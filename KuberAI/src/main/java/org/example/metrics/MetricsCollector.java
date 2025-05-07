package org.example.metrics;

import io.fabric8.kubernetes.api.model.Container;
import io.fabric8.kubernetes.api.model.Pod;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 用于收集Kubernetes集群中Pod资源使用指标的工具类
 * 
 * @author 刘晋勋
 */
public class MetricsCollector {
    
    private static final Logger log = LoggerFactory.getLogger(MetricsCollector.class);
    
    private final KubernetesClient client;
    
    public MetricsCollector() {
        this.client = new KubernetesClientBuilder().build();
    }
    
    public MetricsCollector(KubernetesClient client) {
        this.client = client;
    }
    
    /**
     * 获取指定命名空间中Pod的资源使用情况
     * @param namespace 命名空间
     * @return Map包含Pod名称和资源使用情况
     */
    public Map<String, PodMetrics> collectPodMetrics(String namespace) {
        Map<String, PodMetrics> metricsMap = new HashMap<>();
        log.info("正在收集命名空间 " + namespace + " 中的Pod指标...");
        
        try {
            // 获取命名空间中的所有Pod
            List<Pod> pods = client.pods().inNamespace(namespace).list().getItems();
            log.info("找到 " + pods.size() + " 个Pod在命名空间 " + namespace + " 中");
            
            // 获取Pod的指标数据
            client.top().pods().metrics(namespace).getItems().forEach(podMetrics -> {
                String podName = podMetrics.getMetadata().getName();
                
                AtomicReference<Double> cpuUsage = new AtomicReference<>(0.0);
                AtomicReference<Double> memoryUsage = new AtomicReference<>(0.0);
                
                // 计算每个容器的资源使用情况
                podMetrics.getContainers().forEach(containerMetrics -> {
                    String cpuQuantity = containerMetrics.getUsage().get("cpu").getAmount();
                    String memoryQuantity = containerMetrics.getUsage().get("memory").getAmount();
                    
                    // 将CPU转换为核心数（去掉单位如m）
                    cpuUsage.updateAndGet(v -> v + parseQuantity(cpuQuantity));
                    
                    // 将内存转换为MB（去掉单位如Ki, Mi, Gi）
                    memoryUsage.updateAndGet(v -> v + parseMemoryToMB(memoryQuantity));
                });
                
                // 获取Pod的资源限制和请求
                Pod pod = pods.stream()
                        .filter(p -> p.getMetadata().getName().equals(podName))
                        .findFirst()
                        .orElse(null);
                
                double cpuLimit = 0;
                double memoryLimit = 0;
                double cpuRequest = 0;
                double memoryRequest = 0;
                
                if (pod != null && pod.getSpec() != null && pod.getSpec().getContainers() != null) {
                    for (Container container : pod.getSpec().getContainers()) {
                        if (container.getResources() != null) {
                            // 获取CPU和内存限制
                            if (container.getResources().getLimits() != null) {
                                if (container.getResources().getLimits().get("cpu") != null) {
                                    cpuLimit += parseQuantity(container.getResources().getLimits().get("cpu").getAmount());
                                }
                                if (container.getResources().getLimits().get("memory") != null) {
                                    memoryLimit += parseMemoryToMB(container.getResources().getLimits().get("memory").getAmount());
                                }
                            }
                            
                            // 获取CPU和内存请求
                            if (container.getResources().getRequests() != null) {
                                if (container.getResources().getRequests().get("cpu") != null) {
                                    cpuRequest += parseQuantity(container.getResources().getRequests().get("cpu").getAmount());
                                }
                                if (container.getResources().getRequests().get("memory") != null) {
                                    memoryRequest += parseMemoryToMB(container.getResources().getRequests().get("memory").getAmount());
                                }
                            }
                        }
                    }
                }
                
                PodMetrics metrics = new PodMetrics(
                        podName,
                        cpuUsage.get(),
                        memoryUsage.get(),
                        cpuRequest,
                        memoryRequest,
                        cpuLimit,
                        memoryLimit,
                        Instant.now()
                );
                
                metricsMap.put(podName, metrics);
                log.info("已收集Pod " + podName + " 的指标: " + metrics);
            });
            
            return metricsMap;
        } catch (Exception e) {
            log.error("收集指标时出错: " + e.getMessage(), e);
            return metricsMap;
        }
    }
    
    /**
     * 收集指定Pod在指定时间段内的历史指标
     * @param namespace 命名空间
     * @param podName Pod名称
     * @param duration 时间段
     * @return Pod的历史指标
     */
    public PodHistoricalMetrics collectPodHistoricalMetrics(String namespace, String podName, Duration duration) {
        // 在实际实现中，这里应该访问Prometheus或其他时序数据库来获取历史数据
        // 这里简单模拟，只返回当前的指标作为历史指标
        log.info("正在收集Pod " + podName + " 的历史指标...");
        
        try {
            Map<String, PodMetrics> currentMetrics = collectPodMetrics(namespace);
            PodMetrics metrics = currentMetrics.get(podName);
            
            if (metrics == null) {
                log.warn("未找到Pod " + podName + " 的指标");
                return new PodHistoricalMetrics(podName, List.of());
            }
            
            return new PodHistoricalMetrics(podName, List.of(metrics));
        } catch (Exception e) {
            log.error("收集历史指标时出错: " + e.getMessage(), e);
            return new PodHistoricalMetrics(podName, List.of());
        }
    }
    
    /**
     * 解析CPU数量，将其转换为核心数
     * @param quantity CPU数量字符串（如"100m"表示0.1核）
     * @return 核心数
     */
    private double parseQuantity(String quantity) {
        if (quantity == null || quantity.isEmpty()) {
            return 0;
        }
        
        try {
            if (quantity.endsWith("m")) {
                // 毫核，1000m = 1核
                return Double.parseDouble(quantity.substring(0, quantity.length() - 1)) / 1000;
            } else {
                // 核
                return Double.parseDouble(quantity);
            }
        } catch (NumberFormatException e) {
            log.warn("解析CPU数量时出错: " + quantity);
            return 0;
        }
    }
    
    /**
     * 将内存数量转换为MB
     * @param quantity 内存数量字符串（如"100Mi"表示100MB）
     * @return MB
     */
    private double parseMemoryToMB(String quantity) {
        if (quantity == null || quantity.isEmpty()) {
            return 0;
        }
        
        try {
            if (quantity.endsWith("Ki")) {
                // KiB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 2)) / 1024;
            } else if (quantity.endsWith("Mi")) {
                // MiB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 2));
            } else if (quantity.endsWith("Gi")) {
                // GiB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 2)) * 1024;
            } else if (quantity.endsWith("Ti")) {
                // TiB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 2)) * 1024 * 1024;
            } else if (quantity.endsWith("K") || quantity.endsWith("k")) {
                // KB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 1)) / 1000;
            } else if (quantity.endsWith("M") || quantity.endsWith("m")) {
                // MB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 1));
            } else if (quantity.endsWith("G") || quantity.endsWith("g")) {
                // GB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 1)) * 1000;
            } else if (quantity.endsWith("T") || quantity.endsWith("t")) {
                // TB到MB
                return Double.parseDouble(quantity.substring(0, quantity.length() - 1)) * 1000 * 1000;
            } else {
                // 假设是字节
                return Double.parseDouble(quantity) / (1024 * 1024);
            }
        } catch (NumberFormatException e) {
            log.warn("解析内存数量时出错: " + quantity);
            return 0;
        }
    }
} 