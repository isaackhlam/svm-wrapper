import io
import traceback
import uuid
from typing import Union

from fastapi import FastAPI, File, UploadFile
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/upload")
async def upload(file: UploadFile = File(...), key: str = str(...)):
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
