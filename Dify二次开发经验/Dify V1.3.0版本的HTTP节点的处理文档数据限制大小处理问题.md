### Dify V1.3.0版本的HTTP节点的处理文档数据限制大小处理问题

#### 问题现象

目前的Dify平台工作流的 "HTTP请求节点"的文档处理的大小最大限制是1MB，当文档大小超出1MB时，会出现错误，例如会有如下报错信息：

```reStructuredText
Text size is too large,max size is 1.00MB,but current size is 1.35MB
```



#### 工作流的“HTTP请求节点”的配置和输入信息

- 配置信息

  - 通过PSOST请求发出文档处理： 例如

    ```bash
    http://10.63.132.21:10019/excel_process
    ```

    BODY的参数是：键：file；类型：file；值：first_record（来自上一节点）

  - 上一节点是过滤节点，其输出信息是：

    ```bash
    {
      "result": [
        {
          "dify_model_identity": "__dify__file__",
          "id": null,
          "tenant_id": "53f5ddaa-2365-43a7-8a5b-cbd704ddd610",
          "type": "document",
          "transfer_method": "local_file",
          "remote_url": "/files/12a2f2ab-3bed-457c-b102-cf914f43468d/file-preview?timestamp=1752199576&nonce=4a115a17317bb04cc98666eb8e7de8bc&sign=lCYI1RrRqCMXGxK_clDZMJ06g3cEGDRLS3hu_WC1CzM=",
          "related_id": "12a2f2ab-3bed-457c-b102-cf914f43468d",
          "filename": "VG-09.12-D02 PFMEA（J1）.xlsx",
          "extension": ".xlsx",
          "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          "size": 1321223,
          "url": "/files/12a2f2ab-3bed-457c-b102-cf914f43468d/file-preview?timestamp=1752199583&nonce=0bc4869c3dae97db0ad117895e424fa5&sign=C2GvbbIbRGQb-nDDLCDySTioS0-rznu3_3k6aCBymp0="
        }
      ],
      "first_record": {
        "dify_model_identity": "__dify__file__",
        "id": null,
        "tenant_id": "53f5ddaa-2365-43a7-8a5b-cbd704ddd610",
        "type": "document",
        "transfer_method": "local_file",
        "remote_url": "/files/12a2f2ab-3bed-457c-b102-cf914f43468d/file-preview?timestamp=1752199576&nonce=4a115a17317bb04cc98666eb8e7de8bc&sign=lCYI1RrRqCMXGxK_clDZMJ06g3cEGDRLS3hu_WC1CzM=",
        "related_id": "12a2f2ab-3bed-457c-b102-cf914f43468d",
        "filename": "VG-09.12-D02 PFMEA（J1）.xlsx",
        "extension": ".xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "size": 1321223,
        "url": "/files/12a2f2ab-3bed-457c-b102-cf914f43468d/file-preview?timestamp=1752199583&nonce=2ba6eff6c446c7049e23b65c01810111&sign=9wIK-NHQjlhpN2Dq2O0gvvNBcs-ft-5kNKI7a0i3NUU="
      },
      "last_record": {
        "dify_model_identity": "__dify__file__",
        "id": null,
        "tenant_id": "53f5ddaa-2365-43a7-8a5b-cbd704ddd610",
        "type": "document",
        "transfer_method": "local_file",
        "remote_url": "/files/12a2f2ab-3bed-457c-b102-cf914f43468d/file-preview?timestamp=1752199576&nonce=4a115a17317bb04cc98666eb8e7de8bc&sign=lCYI1RrRqCMXGxK_clDZMJ06g3cEGDRLS3hu_WC1CzM=",
        "related_id": "12a2f2ab-3bed-457c-b102-cf914f43468d",
        "filename": "VG-09.12-D02 PFMEA（J1）.xlsx",
        "extension": ".xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "size": 1321223,
        "url": "/files/12a2f2ab-3bed-457c-b102-cf914f43468d/file-preview?timestamp=1752199583&nonce=1374a45518dea79f3779d9eeb16907b0&sign=AbkqoqpWgAlS1goCg5-F38iSQYmTLIMGpc8_pmEoH8w="
      }
    }
    ```

    

