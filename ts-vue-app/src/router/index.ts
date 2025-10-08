import { createRouter, createWebHistory } from 'vue-router';
import HelloWorld from '../views/HomeView.vue';
import MainLayout from '../layout/MainLayout.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      meta: { layout: MainLayout },
      component: HelloWorld
    },
    {
      path: '/about',
      name: 'about',
      meta: { layout: MainLayout },
      component: () => import('../views/AboutView.vue')
    },
    {
      path: '/result/:jobId',
      name: 'result',
      meta: { layout: MainLayout },
      component: () => import('../views/ResultView.vue')
    },
    {
      path: '/:pathMatch(.*)*',   // 👈 catch all unmatched routes
      name: '404',
      component: () => import('../views/NotFoundView.vue')
    }
  ]
});

export default router;

