#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { CloudEngineerAgentStack } from '../lib/cloud-engineer-agent-stack';

// 创建CDK应用实例
const app = new cdk.App();
// 实例化云工程师代理堆栈
new CloudEngineerAgentStack(app, 'CloudEngineerAgentStack', {
  env: { 
    account: process.env.CDK_DEFAULT_ACCOUNT, // 使用默认AWS账户
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1' // 使用默认区域或us-east-1
  },
});
