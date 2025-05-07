package org.example.metrics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * Pod资源优化器，用于分析Pod指标并提供资源配置建议
 * 
 * @author 刘晋勋
 */
public class PodResourceOptimizer {
    
    private static final Logger log = LoggerFactory.getLogger(PodResourceOptimizer.class);
    
    private final MetricsCollector metricsCollector;
    
    // CPU请求的目标利用率（通常设置为70-80%，为容错和突发负载留出空间）
    private final double targetCpuUtilization = 0.7;
    
    // 内存请求的目标利用率（通常设置为70-80%，为容错和突发负载留出空间）
    private final double targetMemoryUtilization = 0.7;
    
    // CPU限制与请求的比率（通常为1.5-2.0倍）
    private final double cpuLimitRatio = 1.5;
    
    // 内存限制与请求的比率（通常为1.2-1.5倍）
    private final double memoryLimitRatio = 1.2;
    
    public PodResourceOptimizer(MetricsCollector metricsCollector) {
        this.metricsCollector = metricsCollector;
    }
    
    /**
     * 为指定命名空间中的所有Pod生成资源建议
     * @param namespace 命名空间
     * @return 资源建议映射（Pod名称 -> 资源建议）
     */
    public Map<String, ResourceRecommendation> generateRecommendations(String namespace) {
        log.info("为命名空间 " + namespace + " 中的Pod生成资源建议...");
        
        Map<String, ResourceRecommendation> recommendations = new HashMap<>();
        Map<String, PodMetrics> currentMetrics = metricsCollector.collectPodMetrics(namespace);
        
        for (Map.Entry<String, PodMetrics> entry : currentMetrics.entrySet()) {
            String podName = entry.getKey();
            PodMetrics metrics = entry.getValue();
            
            // 收集历史指标（通常会收集更长时间的历史数据，这里简化为使用当前指标）
            PodHistoricalMetrics historicalMetrics = metricsCollector.collectPodHistoricalMetrics(
                    namespace, podName, Duration.ofDays(7));
            
            // 基于历史指标生成资源建议
            ResourceRecommendation recommendation = generateRecommendationForPod(historicalMetrics);
            recommendations.put(podName, recommendation);
            
            log.info("为Pod " + podName + " 生成资源建议: " + recommendation);
        }
        
        return recommendations;
    }
    
    /**
     * 为单个Pod生成资源建议
     * @param historicalMetrics Pod的历史指标
     * @return 资源建议
     */
    public ResourceRecommendation generateRecommendationForPod(PodHistoricalMetrics historicalMetrics) {
        if (historicalMetrics.getMetricsPoints().isEmpty()) {
            log.warn("Pod " + historicalMetrics.getPodName() + " 没有可用的历史指标数据");
            return new ResourceRecommendation(0, 0, 0, 0);
        }
        
        // 使用p95指标作为基准，确保容器在大多数情况下有足够的资源
        double p95CpuUsage = historicalMetrics.getP95CpuUsage();
        double p95MemoryUsage = historicalMetrics.getP95MemoryUsage();
        
        // 计算推荐的CPU请求（p95 CPU使用量 / 目标利用率）
        double recommendedCpuRequest = p95CpuUsage / targetCpuUtilization;
        
        // 计算推荐的内存请求（p95内存使用量 / 目标利用率）
        double recommendedMemoryRequest = p95MemoryUsage / targetMemoryUtilization;
        
        // 计算推荐的CPU限制（请求的cpuLimitRatio倍）
        double recommendedCpuLimit = recommendedCpuRequest * cpuLimitRatio;
        
        // 计算推荐的内存限制（请求的memoryLimitRatio倍）
        double recommendedMemoryLimit = recommendedMemoryRequest * memoryLimitRatio;
        
        // 四舍五入到合理的精度
        recommendedCpuRequest = roundToPrecision(recommendedCpuRequest, 0.01);
        recommendedCpuLimit = roundToPrecision(recommendedCpuLimit, 0.01);
        recommendedMemoryRequest = roundToPrecision(recommendedMemoryRequest, 1);
        recommendedMemoryLimit = roundToPrecision(recommendedMemoryLimit, 1);
        
        return new ResourceRecommendation(
                recommendedCpuRequest,
                recommendedCpuLimit,
                recommendedMemoryRequest,
                recommendedMemoryLimit
        );
    }
    
    /**
     * 将值四舍五入到指定精度
     * @param value 要四舍五入的值
     * @param precision 精度
     * @return 四舍五入后的值
     */
    private double roundToPrecision(double value, double precision) {
        return Math.round(value / precision) * precision;
    }
    
    /**
     * 资源建议类，包含CPU和内存的请求和限制建议
     */
    public static class ResourceRecommendation {
        /**
         * 推荐的CPU请求（核心数）
         */
        private double cpuRequest;
        
        /**
         * 推荐的CPU限制（核心数）
         */
        private double cpuLimit;
        
        /**
         * 推荐的内存请求（MB）
         */
        private double memoryRequest;
        
        /**
         * 推荐的内存限制（MB）
         */
        private double memoryLimit;
        
        /**
         * 无参数构造函数
         */
        public ResourceRecommendation() {
        }
        
        /**
         * 全参数构造函数
         */
        public ResourceRecommendation(double cpuRequest, double cpuLimit, 
                                     double memoryRequest, double memoryLimit) {
            this.cpuRequest = cpuRequest;
            this.cpuLimit = cpuLimit;
            this.memoryRequest = memoryRequest;
            this.memoryLimit = memoryLimit;
        }
        
        // Getter方法
        public double getCpuRequest() {
            return cpuRequest;
        }
        
        public double getCpuLimit() {
            return cpuLimit;
        }
        
        public double getMemoryRequest() {
            return memoryRequest;
        }
        
        public double getMemoryLimit() {
            return memoryLimit;
        }
        
        // Setter方法
        public void setCpuRequest(double cpuRequest) {
            this.cpuRequest = cpuRequest;
        }
        
        public void setCpuLimit(double cpuLimit) {
            this.cpuLimit = cpuLimit;
        }
        
        public void setMemoryRequest(double memoryRequest) {
            this.memoryRequest = memoryRequest;
        }
        
        public void setMemoryLimit(double memoryLimit) {
            this.memoryLimit = memoryLimit;
        }
        
        /**
         * 将CPU请求转换为Kubernetes资源格式
         * @return Kubernetes资源格式的CPU请求
         */
        public String getCpuRequestString() {
            if (cpuRequest < 1) {
                // 小于1核时，使用毫核表示
                return Math.round(cpuRequest * 1000) + "m";
            } else {
                // 大于等于1核时，直接使用核数表示
                return String.valueOf(cpuRequest);
            }
        }
        
        /**
         * 将CPU限制转换为Kubernetes资源格式
         * @return Kubernetes资源格式的CPU限制
         */
        public String getCpuLimitString() {
            if (cpuLimit < 1) {
                // 小于1核时，使用毫核表示
                return Math.round(cpuLimit * 1000) + "m";
            } else {
                // 大于等于1核时，直接使用核数表示
                return String.valueOf(cpuLimit);
            }
        }
        
        /**
         * 将内存请求转换为Kubernetes资源格式
         * @return Kubernetes资源格式的内存请求
         */
        public String getMemoryRequestString() {
            if (memoryRequest < 1) {
                // 小于1MB时，使用KB表示
                return Math.round(memoryRequest * 1024) + "Ki";
            } else if (memoryRequest < 1024) {
                // 小于1GB时，使用MB表示
                return Math.round(memoryRequest) + "Mi";
            } else {
                // 大于等于1GB时，使用GB表示
                return (memoryRequest / 1024) + "Gi";
            }
        }
        
        /**
         * 将内存限制转换为Kubernetes资源格式
         * @return Kubernetes资源格式的内存限制
         */
        public String getMemoryLimitString() {
            if (memoryLimit < 1) {
                // 小于1MB时，使用KB表示
                return Math.round(memoryLimit * 1024) + "Ki";
            } else if (memoryLimit < 1024) {
                // 小于1GB时，使用MB表示
                return Math.round(memoryLimit) + "Mi";
            } else {
                // 大于等于1GB时，使用GB表示
                return (memoryLimit / 1024) + "Gi";
            }
        }
        
        @Override
        public String toString() {
            return "ResourceRecommendation{" +
                    "cpuRequest=" + cpuRequest +
                    ", cpuLimit=" + cpuLimit +
                    ", memoryRequest=" + memoryRequest +
                    ", memoryLimit=" + memoryLimit +
                    '}';
        }
    }
} 