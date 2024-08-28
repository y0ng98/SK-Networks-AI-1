import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'

import BoardRoutes from '@/board/router/BoardRoutes'
import ProductRoutes from '@/product/router/ProductRoutes'
import HomeRoutes from '@/home/router/HomeRoutes'
import AccountRoutes from '@/account/router/AccountRoutes'
import AuthenticationRoutes from '@/authentication/router/AuthenticationRoutes'
import LogisticRegressionRoutes from '@/logisticRegression/router/LogisticRegressionRoutes'
import TrainTestEvaluationRoutes from '@/trainTestEvaluation/router/TrainTestEvaluationRoutes'
import PolynomialRegressionRoutes from '@/polynomialRegression/router/PolynomialRegressionRoutes'
import ExponentialRegressionRoutes from '@/exponentialRegression/router/ExponentialRegressionRoutes'
import RandomForestRoutes from '@/randomForest/router/RandomForestRoutes'
import PostRoutes from '@/post/router/PostRoutes'
import KmeansRoutes from '@/kmeans/router/KmeansRoutes'
import TensorFlowIrisTestRoutes from '@/tfIris/router/TensorFlowIrisTestRoutes'
import CartRoutes from '@/cart/router/CartRoutes'
import PrincipalComponentAnalysisRoutes from '@/principalComponentAnalysis/router/PrincipalComponentAnalysisRoutes'
import KafkaTestRoutes from '@/kafka/router/KafkaTestRoutes'
import GatherEverythingRoutes from '@/gatherEverything/router/GatherEverythingRoutes'
import FileS3TestRoutes from '@/fileS3/router/FileS3TestRoutes'

const routes: Array<RouteRecordRaw> = [
  ...HomeRoutes,
  ...BoardRoutes,
  ...ProductRoutes,
  ...AccountRoutes,
  ...AuthenticationRoutes,
  ...LogisticRegressionRoutes,
  ...TrainTestEvaluationRoutes,
  ...PolynomialRegressionRoutes,
  ...ExponentialRegressionRoutes,
  ...RandomForestRoutes,
  ...PostRoutes,
  ...KmeansRoutes,
  ...TensorFlowIrisTestRoutes,
  ...CartRoutes,
  ...PrincipalComponentAnalysisRoutes,
  ...KafkaTestRoutes,
  ...GatherEverythingRoutes,
  ...FileS3TestRoutes,
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
