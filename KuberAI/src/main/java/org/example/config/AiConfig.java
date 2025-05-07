package org.example.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.beans.factory.annotation.Value;

/**
 * Spring AI 配置类
 * 提供通义千问模型访问配置
 */
@Configuration
public class AiConfig {

    @Value("${spring.ai.tongyi.api-key:}")
    private String apiKey;

    /**
     * Kubernetes资源分析系统提示模板
     */
    @Bean
    public PromptTemplate kubernetesPromptTemplate() {
        return new PromptTemplate("""
            你是一位Kubernetes资源优化专家，擅长分析和优化容器化应用的资源配置。
            你需要根据提供的指标数据，提供专业、详细且具体的资源配置建议。
            请确保考虑以下几点：
            1. 资源使用率和效率最大化
            2. 成本优化，避免过度分配资源
            3. 应用稳定性，避免资源不足造成的问题
            4. Kubernetes最佳实践
            5. 提供具体的配置示例
            
            当用户提出Kubernetes资源相关问题时，请提供详细的分析和优化建议。
            """);
    }

    @Bean
    public ChatClient chatClient(ChatClient.Builder builder) {
        return builder.build();
    }
} 