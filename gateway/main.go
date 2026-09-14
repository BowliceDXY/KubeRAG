package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ===== 配置（支持环境变量覆盖）=====
const (
	defaultGatewayPort = ":8080"
	defaultBackendURL  = "http://127.0.0.1:9003" // Python FastAPI 服务地址（统一 9003）
	rateLimit          = 10                       // 每个 IP 每秒最多 10 个请求
	ipIdleTimeout      = 10 * time.Minute         // IP 记录空闲超时
	cleanupInterval    = 5 * time.Minute          // 空闲 IP 清理协程运行间隔
)

// ===== 令牌桶限流器 =====
type ipLimiter struct {
	mu       sync.Mutex
	tokens   map[string]int
	lastTime map[string]time.Time
}

func newIPLimiter() *ipLimiter {
	return &ipLimiter{
		tokens:   make(map[string]int),
		lastTime: make(map[string]time.Time),
	}
}

func (l *ipLimiter) allow(ip string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now()
	last, exists := l.lastTime[ip]
	if !exists {
		l.tokens[ip] = rateLimit
		l.lastTime[ip] = now
		return true
	}

	// 按时间补充令牌
	elapsed := now.Sub(last).Seconds()
	newTokens := int(elapsed * float64(rateLimit))
	if newTokens > 0 {
		l.tokens[ip] = min(l.tokens[ip]+newTokens, rateLimit)
		l.lastTime[ip] = now
	}

	if l.tokens[ip] > 0 {
		l.tokens[ip]--
		return true
	}
	return false
}

// cleanupLoop 定期清理长时间未活跃的 IP 记录，防止限流 map 无限增长（内存泄漏）
func (l *ipLimiter) cleanupLoop() {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()
	for range ticker.C {
		l.mu.Lock()
		cutoff := time.Now().Add(-ipIdleTimeout)
		for ip, last := range l.lastTime {
			if last.Before(cutoff) {
				delete(l.lastTime, ip)
				delete(l.tokens, ip)
			}
		}
		l.mu.Unlock()
	}
}

var limiter = newIPLimiter()

// ===== 日志中间件 =====
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(ww, r)
		log.Printf("[%s] %s %s -> %d (%v)", r.RemoteAddr, r.Method, r.URL.Path, ww.status, time.Since(start))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

// ===== 限流中间件 =====
func rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := r.RemoteAddr
		if !limiter.allow(ip) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			json.NewEncoder(w).Encode(map[string]string{"error": "请求过于频繁，请稍后再试"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ===== 反向代理：转发到 Python 服务 =====
func proxyHandler(backendURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		targetURL := backendURL + r.URL.Path
		if r.URL.RawQuery != "" {
			targetURL += "?" + r.URL.RawQuery
		}

		req, err := http.NewRequest(r.Method, targetURL, r.Body)
		if err != nil {
			http.Error(w, "创建请求失败", http.StatusInternalServerError)
			return
		}

		// 复制请求头
		for key, values := range r.Header {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}

		client := &http.Client{Timeout: 120 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			json.NewEncoder(w).Encode(map[string]string{"error": "后端服务不可达，请确认 Python 服务已启动"})
			return
		}
		defer resp.Body.Close()

		// 复制响应头
		for key, values := range resp.Header {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
		w.WriteHeader(resp.StatusCode)

		// 流式复制响应体（支持 /ask/stream 的逐字输出）
		io.Copy(w, resp.Body)
	}
}

// ===== 网关健康检查 =====
func healthHandler(backendURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "ok",
			"service": "KubeRAG Gateway",
			"backend": backendURL,
		})
	}
}

func main() {
	// 配置：环境变量优先，缺省用统一默认值
	gatewayPort := os.Getenv("GATEWAY_PORT")
	if gatewayPort == "" {
		gatewayPort = defaultGatewayPort
	}
	if !strings.HasPrefix(gatewayPort, ":") {
		gatewayPort = ":" + gatewayPort
	}
	backendURL := os.Getenv("BACKEND_URL")
	if backendURL == "" {
		backendURL = defaultBackendURL
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler(backendURL))
	mux.HandleFunc("/", proxyHandler(backendURL)) // 其他请求全部转发到 Python

	// 中间件链：限流 → 日志 → 路由
	handler := rateLimitMiddleware(loggingMiddleware(mux))

	// 启动空闲 IP 清理协程，防止限流 map 无限增长
	go limiter.cleanupLoop()

	fmt.Printf("========================================\n")
	fmt.Printf("  KubeRAG Go 网关启动\n")
	fmt.Printf("  网关地址: http://127.0.0.1%s\n", gatewayPort)
	fmt.Printf("  后端服务: %s\n", backendURL)
	fmt.Printf("  限流: 每IP每秒%d个请求\n", rateLimit)
	fmt.Printf("========================================\n")

	log.Fatal(http.ListenAndServe(gatewayPort, handler))
}
