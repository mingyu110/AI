package org.example.metrics;

import java.time.Instant;

/**
 * 表示Pod的资源使用指标
 * 
 * @author 刘晋勋
 */
public class PodMetrics {
    
    /**
     * Pod名称
     */
    private String podName;
    
    /**
     * CPU使用量（核心数）
     */
    private double cpuUsage;
    
    /**
     * 内存使用量（MB）
     */
    private double memoryUsage;
    
    /**
     * CPU请求（核心数）
     */
    private double cpuRequest;
    
    /**
     * 内存请求（MB）
     */
    private double memoryRequest;
    
    /**
     * CPU限制（核心数）
     */
    private double cpuLimit;
    
    /**
     * 内存限制（MB）
     */
    private double memoryLimit;
    
    /**
     * 指标收集时间
     */
    private Instant timestamp;
    
    /**
     * 无参构造函数
     */
    public PodMetrics() {
    }
    
    /**
     * 全参数构造函数
     */
    public PodMetrics(String podName, double cpuUsage, double memoryUsage, double cpuRequest, 
                     double memoryRequest, double cpuLimit, double memoryLimit, Instant timestamp) {
        this.podName = podName;
        this.cpuUsage = cpuUsage;
        this.memoryUsage = memoryUsage;
        this.cpuRequest = cpuRequest;
        this.memoryRequest = memoryRequest;
        this.cpuLimit = cpuLimit;
        this.memoryLimit = memoryLimit;
        this.timestamp = timestamp;
    }
    
    // Getter方法
    public String getPodName() {
        return podName;
    }
    
    public double getCpuUsage() {
        return cpuUsage;
    }
    
    public double getMemoryUsage() {
        return memoryUsage;
    }
    
    public double getCpuRequest() {
        return cpuRequest;
    }
    
    public double getMemoryRequest() {
        return memoryRequest;
    }
    
    public double getCpuLimit() {
        return cpuLimit;
    }
    
    public double getMemoryLimit() {
        return memoryLimit;
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
    
    // Setter方法
    public void setPodName(String podName) {
        this.podName = podName;
    }
    
    public void setCpuUsage(double cpuUsage) {
        this.cpuUsage = cpuUsage;
    }
    
    public void setMemoryUsage(double memoryUsage) {
        this.memoryUsage = memoryUsage;
    }
    
    public void setCpuRequest(double cpuRequest) {
        this.cpuRequest = cpuRequest;
    }
    
    public void setMemoryRequest(double memoryRequest) {
        this.memoryRequest = memoryRequest;
    }
    
    public void setCpuLimit(double cpuLimit) {
        this.cpuLimit = cpuLimit;
    }
    
    public void setMemoryLimit(double memoryLimit) {
        this.memoryLimit = memoryLimit;
    }
    
    public void setTimestamp(Instant timestamp) {
        this.timestamp = timestamp;
    }
    
    /**
     * 计算CPU利用率（使用量/请求）
     * @return CPU利用率
     */
    public double getCpuUtilizationRatio() {
        if (cpuRequest <= 0) {
            return 0;
        }
        return cpuUsage / cpuRequest;
    }
    
    /**
     * 计算内存利用率（使用量/请求）
     * @return 内存利用率
     */
    public double getMemoryUtilizationRatio() {
        if (memoryRequest <= 0) {
            return 0;
        }
        return memoryUsage / memoryRequest;
    }
    
    /**
     * 计算CPU使用率（使用量/限制）
     * @return CPU使用率
     */
    public double getCpuUsageRatio() {
        if (cpuLimit <= 0) {
            return 0;
        }
        return cpuUsage / cpuLimit;
    }
    
    /**
     * 计算内存使用率（使用量/限制）
     * @return 内存使用率
     */
    public double getMemoryUsageRatio() {
        if (memoryLimit <= 0) {
            return 0;
        }
        return memoryUsage / memoryLimit;
    }
    
    @Override
    public String toString() {
        return "PodMetrics{" +
                "podName='" + podName + '\'' +
                ", cpuUsage=" + cpuUsage +
                ", memoryUsage=" + memoryUsage +
                ", cpuRequest=" + cpuRequest +
                ", memoryRequest=" + memoryRequest +
                ", cpuLimit=" + cpuLimit +
                ", memoryLimit=" + memoryLimit +
                ", timestamp=" + timestamp +
                '}';
    }
} 