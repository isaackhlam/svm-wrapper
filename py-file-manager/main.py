import io
import traceback
import uuid
from typing import Union

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from minio import Minio

client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

# Ensure a bucket exists
bucket_name = "my-bucket"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created")
else:
    print(f"Bucket '{bucket_name}' already exists")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # or specify ["POST", "GET"]
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/upload")
async def upload(file: UploadFile = File(...), key: str = Form(...)):
    try:
        fileobj = io.BytesIO(file.file.read())
        # TODO: Parse file extension with key
        # key = str(uuid.uuid4())
        result = client.put_object(
            bucket_name,
            key,
            fileobj,
            length=-1,
            part_size=10 * 1024 * 1024,
        )
        print(f"appended {result.object_name} object; etag: {result.etag}")
    except Exception as ex:
        print(traceback.format_exc())
        return {
            "statusCode": 400,
            "body": ex,
        }


@app.get("/download/{key}")
async def download(key: str):
    filename = f"/{key}/output.csv"
    try:
        obj = client.get_object(
            bucket_name,
            filename,
        )
        stat = client.stat_object(bucket_name, filename)
        content_type = stat.content_type or "application/octet-stream"

        return StreamingResponse(
            obj,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{key}_output.csv"'},
        )
    except Exception as ex:
        print(traceback.format_exc())
        return {
            "statusCode": 400,
            "body": ex,
        }


@app.get("/download/shap/{key}")
async def download_shap(key: str):
    filename = f"{key}/shap.csv"
    try:
        obj = client.get_object(
            bucket_name,
            filename,
        )
        stat = client.stat_object(bucket_name, filename)
        content_type = stat.content_type or "application/octet-stream"

        return StreamingResponse(
            obj,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{key}"'},
        )
    except Exception as ex:
        print(traceback.format_exc())
        return {
            "statusCode": 400,
            "body": ex,
        }
