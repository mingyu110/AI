package org.example.chat;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Kubernetes资源优化推荐控制器
 * 通过Spring AI Alibaba提供智能化的资源分配建议
 * 
 * @author 刘晋勋
 */
@RestController
@RequestMapping("/api/resources")
public class ResourceRecommendationController {

    private final ResourceRecommendationService recommendationService;
    
    /**
     * 构造函数注入ResourceRecommendationService
     * @param recommendationService 资源推荐服务
     */
    public ResourceRecommendationController(ResourceRecommendationService recommendationService) {
        this.recommendationService = recommendationService;
    }

    /**
     * 获取命名空间内所有Pod的资源优化建议
     * @param namespace Kubernetes命名空间
     * @return 通过AI生成的资源优化建议
     */
    @GetMapping("/namespace")
    public String getNamespaceResourceRecommendations(@RequestParam String namespace) {
        return recommendationService.generateResourceRecommendations(namespace);
    }

    /**
     * 获取特定Pod的资源优化建议
     * @param namespace Kubernetes命名空间
     * @param podName Pod名称
     * @return 通过AI生成的Pod资源优化建议
     */
    @GetMapping("/pod")
    public String getPodResourceRecommendation(
            @RequestParam String namespace, 
            @RequestParam String podName) {
        return recommendationService.generatePodResourceRecommendation(namespace, podName);
    }
} 