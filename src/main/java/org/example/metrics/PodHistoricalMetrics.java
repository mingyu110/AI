package org.example.metrics;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalDouble;

/**
 * 表示Pod的历史资源使用指标
 * 
 * @author 刘晋勋
 */
public class PodHistoricalMetrics {
    
    /**
     * Pod名称
     */
    private String podName;
    
    /**
     * 历史指标数据点列表
     */
    private List<PodMetrics> metricsPoints = new ArrayList<>();
    
    /**
     * 无参构造函数
     */
    public PodHistoricalMetrics() {
    }
    
    /**
     * 全参数构造函数
     */
    public PodHistoricalMetrics(String podName, List<PodMetrics> metricsPoints) {
        this.podName = podName;
        this.metricsPoints = metricsPoints;
    }
    
    // Getter方法
    public String getPodName() {
        return podName;
    }
    
    public List<PodMetrics> getMetricsPoints() {
        return metricsPoints;
    }
    
    // Setter方法
    public void setPodName(String podName) {
        this.podName = podName;
    }
    
    public void setMetricsPoints(List<PodMetrics> metricsPoints) {
        this.metricsPoints = metricsPoints;
    }
    
    /**
     * 获取CPU使用量的平均值
     * @return CPU使用量平均值（核心数）
     */
    public double getAverageCpuUsage() {
        OptionalDouble avg = metricsPoints.stream()
                .mapToDouble(pod -> pod.getCpuUsage())
                .average();
        return avg.orElse(0);
    }
    
    /**
     * 获取内存使用量的平均值
     * @return 内存使用量平均值（MB）
     */
    public double getAverageMemoryUsage() {
        OptionalDouble avg = metricsPoints.stream()
                .mapToDouble(pod -> pod.getMemoryUsage())
                .average();
        return avg.orElse(0);
    }
    
    /**
     * 获取CPU使用量的峰值
     * @return CPU使用量峰值（核心数）
     */
    public double getPeakCpuUsage() {
        OptionalDouble max = metricsPoints.stream()
                .mapToDouble(pod -> pod.getCpuUsage())
                .max();
        return max.orElse(0);
    }
    
    /**
     * 获取内存使用量的峰值
     * @return 内存使用量峰值（MB）
     */
    public double getPeakMemoryUsage() {
        OptionalDouble max = metricsPoints.stream()
                .mapToDouble(pod -> pod.getMemoryUsage())
                .max();
        return max.orElse(0);
    }
    
    /**
     * 获取CPU使用率的95百分位
     * @return CPU使用率的95百分位（核心数）
     */
    public double getP95CpuUsage() {
        if (metricsPoints.isEmpty()) {
            return 0;
        }
        
        List<Double> cpuUsages = metricsPoints.stream()
                .map(pod -> pod.getCpuUsage())
                .sorted()
                .toList();
        
        int index = (int) Math.ceil(cpuUsages.size() * 0.95) - 1;
        if (index < 0) {
            index = 0;
        }
        return cpuUsages.get(index);
    }
    
    /**
     * 获取内存使用率的95百分位
     * @return 内存使用率的95百分位（MB）
     */
    public double getP95MemoryUsage() {
        if (metricsPoints.isEmpty()) {
            return 0;
        }
        
        List<Double> memoryUsages = metricsPoints.stream()
                .map(pod -> pod.getMemoryUsage())
                .sorted()
                .toList();
        
        int index = (int) Math.ceil(memoryUsages.size() * 0.95) - 1;
        if (index < 0) {
            index = 0;
        }
        return memoryUsages.get(index);
    }
    
    /**
     * 获取当前CPU请求
     * @return 当前CPU请求（核心数）
     */
    public double getCurrentCpuRequest() {
        if (metricsPoints.isEmpty()) {
            return 0;
        }
        return metricsPoints.get(metricsPoints.size() - 1).getCpuRequest();
    }
    
    /**
     * 获取当前内存请求
     * @return 当前内存请求（MB）
     */
    public double getCurrentMemoryRequest() {
        if (metricsPoints.isEmpty()) {
            return 0;
        }
        return metricsPoints.get(metricsPoints.size() - 1).getMemoryRequest();
    }
    
    /**
     * 获取当前CPU限制
     * @return 当前CPU限制（核心数）
     */
    public double getCurrentCpuLimit() {
        if (metricsPoints.isEmpty()) {
            return 0;
        }
        return metricsPoints.get(metricsPoints.size() - 1).getCpuLimit();
    }
    
    /**
     * 获取当前内存限制
     * @return 当前内存限制（MB）
     */
    public double getCurrentMemoryLimit() {
        if (metricsPoints.isEmpty()) {
            return 0;
        }
        return metricsPoints.get(metricsPoints.size() - 1).getMemoryLimit();
    }
    
    @Override
    public String toString() {
        return "PodHistoricalMetrics{" +
                "podName='" + podName + '\'' +
                ", metricsPoints=" + metricsPoints +
                '}';
    }
} 