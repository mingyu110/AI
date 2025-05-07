package org.example.chat;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.example.metrics.MetricsCollector;
import org.example.metrics.PodHistoricalMetrics;
import org.example.metrics.PodResourceOptimizer;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

/**
 * 基于Spring AI的资源推荐服务，用于生成Kubernetes资源优化建议
 * 
 * @author 刘晋勋
 */
@Service
public class ResourceRecommendationService {

    private static final Logger log = LoggerFactory.getLogger(ResourceRecommendationService.class);
    
    private final ChatClient chatClient;
    
    /**
     * 构造函数注入ChatClient
     * @param chatClient Spring AI的ChatClient
     */
    public ResourceRecommendationService(ChatClient chatClient) {
        this.chatClient = chatClient;
    }
    
    /**
     * 获取命名空间中所有Pod的资源配置建议
     * @param namespace 命名空间名称
     * @return 大模型生成的资源配置建议
     */
    public String generateResourceRecommendations(String namespace) {
        log.info("正在为命名空间 " + namespace + " 中的Pod提供资源优化建议...");
        
        try {
            MetricsCollector metricsCollector = new MetricsCollector();
            PodResourceOptimizer optimizer = new PodResourceOptimizer(metricsCollector);
            
            // 生成资源建议
            Map<String, PodResourceOptimizer.ResourceRecommendation> recommendations = 
                    optimizer.generateRecommendations(namespace);
            
            // 构建建议文本
            StringBuilder recommendationsText = new StringBuilder();
            recommendationsText.append("## " + namespace + " 命名空间Pod资源建议\n\n");
            
            recommendations.forEach((podName, recommendation) -> {
                recommendationsText.append("### Pod: " + podName + "\n");
                recommendationsText.append("- 推荐CPU请求: " + recommendation.getCpuRequestString() + "\n");
                recommendationsText.append("- 推荐CPU限制: " + recommendation.getCpuLimitString() + "\n");
                recommendationsText.append("- 推荐内存请求: " + recommendation.getMemoryRequestString() + "\n");
                recommendationsText.append("- 推荐内存限制: " + recommendation.getMemoryLimitString() + "\n\n");
            });
            
            // 使用大模型生成优化建议
            String systemPromptTemplate = """
                你是一位Kubernetes资源优化专家，负责分析Pod资源使用情况并提供优化建议。
                你需要根据提供的指标数据，为用户生成专业、详细的Kubernetes资源配置建议。
                请确保你的建议既有技术深度，又容易理解，包含解释与最佳实践。
                """;
            
            String userPromptTemplate = "以下是{namespace}命名空间中Pod的资源使用情况和建议配置，请基于这些数据为用户提供专业的资源优化建议：\n\n{recommendations}";
            
            Map<String, Object> params = new HashMap<>();
            params.put("namespace", namespace);
            params.put("recommendations", recommendationsText.toString());
            
            SystemPromptTemplate systemPrompt = new SystemPromptTemplate(systemPromptTemplate);
            PromptTemplate userPrompt = new PromptTemplate(userPromptTemplate, params);
            
            return this.chatClient.prompt()
                .system(systemPrompt.render())
                .user(userPrompt.render(params))
                .call()
                .content();
            
        } catch (Exception e) {
            log.error("获取资源建议时出错: " + e.getMessage(), e);
            return "生成资源建议时发生错误: " + e.getMessage();
        }
    }
    
    /**
     * 获取特定Pod的资源配置建议
     * @param namespace 命名空间名称
     * @param podName Pod名称
     * @return 大模型生成的资源配置建议
     */
    public String generatePodResourceRecommendation(String namespace, String podName) {
        log.info("正在为Pod " + podName + " 在命名空间 " + namespace + " 中提供资源优化建议...");
        
        try {
            MetricsCollector metricsCollector = new MetricsCollector();
            PodResourceOptimizer optimizer = new PodResourceOptimizer(metricsCollector);
            
            // 收集Pod的历史指标
            PodHistoricalMetrics historicalMetrics = metricsCollector.collectPodHistoricalMetrics(namespace, podName, java.time.Duration.ofDays(7));
            
            // 生成资源建议
            PodResourceOptimizer.ResourceRecommendation recommendation = optimizer.generateRecommendationForPod(historicalMetrics);
            
            // 构建建议文本
            StringBuilder recommendationText = new StringBuilder();
            recommendationText.append("## Pod资源使用分析: " + podName + "\n\n");
            recommendationText.append("### 当前资源使用情况\n");
            recommendationText.append("- 平均CPU使用量: " + historicalMetrics.getAverageCpuUsage() + "核\n");
            recommendationText.append("- P95 CPU使用量: " + historicalMetrics.getP95CpuUsage() + "核\n");
            recommendationText.append("- 平均内存使用量: " + historicalMetrics.getAverageMemoryUsage() + "MB\n");
            recommendationText.append("- P95 内存使用量: " + historicalMetrics.getP95MemoryUsage() + "MB\n\n");
            
            recommendationText.append("### 当前资源配置\n");
            recommendationText.append("- 当前CPU请求: " + historicalMetrics.getCurrentCpuRequest() + "核\n");
            recommendationText.append("- 当前CPU限制: " + historicalMetrics.getCurrentCpuLimit() + "核\n");
            recommendationText.append("- 当前内存请求: " + historicalMetrics.getCurrentMemoryRequest() + "MB\n");
            recommendationText.append("- 当前内存限制: " + historicalMetrics.getCurrentMemoryLimit() + "MB\n\n");
            
            recommendationText.append("### 资源优化建议\n");
            recommendationText.append("- 推荐CPU请求: " + recommendation.getCpuRequestString() + "\n");
            recommendationText.append("- 推荐CPU限制: " + recommendation.getCpuLimitString() + "\n");
            recommendationText.append("- 推荐内存请求: " + recommendation.getMemoryRequestString() + "\n");
            recommendationText.append("- 推荐内存限制: " + recommendation.getMemoryLimitString() + "\n");

            // 使用大模型生成优化建议
            String systemPromptTemplate = """
                你是一位Kubernetes资源优化专家，负责分析单个Pod的资源使用情况并提供优化建议。
                请根据提供的指标数据，为用户生成专业、详细的Kubernetes资源配置建议。
                你的建议应该包括:
                1. 当前资源使用分析
                2. 潜在的资源浪费或不足问题
                3. 具体的优化配置及原因
                4. 资源调整后可能的收益
                5. 相关的Kubernetes最佳实践
                """;
            
            String userPromptTemplate = "以下是Pod {podName} 在命名空间 {namespace} 的资源使用情况和建议配置：\n\n{recommendations}\n\n请基于这些数据提供专业的资源优化建议和具体的yaml配置示例。";
            
            Map<String, Object> params = new HashMap<>();
            params.put("namespace", namespace);
            params.put("podName", podName);
            params.put("recommendations", recommendationText.toString());
            
            SystemPromptTemplate systemPrompt = new SystemPromptTemplate(systemPromptTemplate);
            PromptTemplate userPrompt = new PromptTemplate(userPromptTemplate, params);
            
            return this.chatClient.prompt()
                .system(systemPrompt.render())
                .user(userPrompt.render(params))
                .call()
                .content();
            
        } catch (Exception e) {
            log.error("获取Pod资源建议时出错: " + e.getMessage(), e);
            return "生成Pod资源建议时发生错误: " + e.getMessage();
        }
    }
} 