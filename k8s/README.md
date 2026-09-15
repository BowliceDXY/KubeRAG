# KubeRAG Kubernetes 部署

将 KubeRAG 部署到 Kubernetes 集群的完整 manifest 和操作指南。

## 架构

```
                    ┌─────────────┐
  外部流量 ────────▶│   Ingress   │
                    └──────┬──────┘
                           │ :8080
                    ┌──────▼──────┐
                    │  Gateway ×2 │ (HPA: 2-3)
                    └──────┬──────┘
                           │ :9003
                    ┌──────▼──────┐
                    │  Backend ×1 │ (固定副本：本地文件存储单写者)
                    └──┬───────┬──┘
                       │       │
                ┌──────▼─┐   ┌─▼──────┐
                │ Redis  │   │  PVC   │
                │  ×1    │   │ 5Gi    │
                └────────┘   └────────┘
```

## 前置条件

- 可用的 Kubernetes 集群（minikube / kind / 云厂商均可）
- `kubectl` 已配置
- [metrics-server](https://github.com/kubernetes-sigs/metrics-server)（HPA 必需）
- [ingress-nginx](https://kubernetes.github.io/ingress-nginx/)（Ingress 必需，可选）

## 1. 构建并推送镜像

```bash
# 后端镜像
docker build -t kuberag-backend:latest .
docker tag kuberag-backend:latest <your-registry>/kuberag-backend:latest
docker push <your-registry>/kuberag-backend:latest

# 网关镜像
docker build -t kuberag-gateway:latest -f gateway/Dockerfile .
docker tag kuberag-gateway:latest <your-registry>/kuberag-gateway:latest
docker push <your-registry>/kuberag-gateway:latest
```

> 如果使用本地集群（minikube/kind），可跳过推送，直接 `eval $(minikube docker-env)` 后构建，镜像在集群内可用。

## 2. 配置密钥

编辑 `secret.yaml`，填入真实 API Key：

```bash
# 或者用 kubectl 创建（更安全，不写入文件）
kubectl create secret generic kuberag-secrets -n kuberag \
  --from-literal=DEEPSEEK_API_KEY=your-key \
  --from-literal=BIGMODEL_API_KEY=your-key \
  --from-literal=TAVILY_API_KEY=your-key
```

## 3. 部署

```bash
# 按顺序应用所有 manifest
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/gateway.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml   # 可选，需要 ingress-nginx
```

或者一键部署：
```bash
kubectl apply -f k8s/
```

## 4. 验证

```bash
# 查看命名空间下所有资源
kubectl get all -n kuberag

# 查看 Pod 状态
kubectl get pods -n kuberag -w

# 查看日志
kubectl logs -f deployment/backend -n kuberag
kubectl logs -f deployment/gateway -n kuberag

# 查看 HPA 状态
kubectl get hpa -n kuberag

# 健康检查（端口转发）
kubectl port-forward svc/gateway -n kuberag 8080:8080
curl http://localhost:8080/health
```

## 5. 访问应用

**方式一：Ingress（推荐）**
```bash
# 配置 hosts 解析（本地测试）
echo "$(minikube ip) kuberag.local" | sudo tee -a /etc/hosts
# 浏览器访问 http://kuberag.local
```

**方式二：端口转发**
```bash
kubectl port-forward svc/gateway -n kuberag 8080:8080
# 浏览器访问 http://localhost:8080
```

**方式三：LoadBalancer（云集群）**
将 `gateway.yaml` 中 Service 的 `type` 改为 `LoadBalancer`，获取外部 IP 访问。

## 6. 清理

```bash
kubectl delete namespace kuberag
```

## 生产环境建议

- Redis：使用托管服务（AWS ElastiCache / 阿里云 Redis）或 Redis Operator，而非单实例
- 镜像：使用固定版本 tag 而非 `latest`，配合 ImagePullPolicy=Always
- 密钥：使用 ExternalSecret / SealedSecret / Vault 管理，不直接提交 Secret 文件
- 持久化：ChromaDB 与 SQLite 都是本地文件单写者存储，因此 Backend 当前固定 1 副本（HPA 仅作用于无状态的 Gateway）。要支持 Backend 多副本扩容，需先把向量库迁移到独立服务（如 Qdrant / Weaviate）、对话历史迁移到共享数据库（如 PostgreSQL），再开启 Backend HPA
- 监控：接入 Prometheus + Grafana，配置 SLO 告警
- CI/CD：使用 ArgoCD / Flux 实现 GitOps 部署
