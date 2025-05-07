package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.ai.chat.client.ChatClient;

/**
 * KuberAI应用主类
 * 提供基于Spring AI Alibaba的Kubernetes资源智能优化功能
 * 
 * @author 刘晋勋
 */
@SpringBootApplication
@RestController
public class Main {

    private final ChatClient chatClient;
    
    /**
     * 构造函数注入ChatClient
     * @param chatClient Spring AI的ChatClient实例
     */
    public Main(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    public static void main(String[] args) {
        SpringApplication.run(Main.class, args);
    }

    @GetMapping("/")
    public String home() {
        return "KuberAI: 基于Spring AI Alibaba的Kubernetes资源智能优化系统";
    }

    @GetMapping("/chat")
    public String chat(@RequestParam String input) {
        return this.chatClient.prompt()
            .user(input)
            .call()
            .content();
    }

    @GetMapping("/resource-optimization")
    public String resourceOptimization(@RequestParam String namespace, @RequestParam(required = false) String podName) {
        String prompt;
        if (podName != null && !podName.isEmpty()) {
            prompt = String.format("优化%s命名空间中的%s资源配置", namespace, podName);
        } else {
            prompt = String.format("分析%s命名空间中的所有Pod，并提供资源配置建议", namespace);
        }

        return this.chatClient.prompt()
            .user(prompt)
            .call()
            .content();
    }
} 