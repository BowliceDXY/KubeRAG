# KubeRAG Go 网关镜像（多阶段构建：编译 + 精简运行）
FROM golang:1.23-alpine AS builder
WORKDIR /build
COPY go.mod ./
COPY main.go ./
RUN go build -o gateway .

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /build/gateway .
EXPOSE 8080
CMD ["./gateway"]
