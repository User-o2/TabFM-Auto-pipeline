# TabFM-Auto：TabArena 最终 Pipeline 代码与 Write-up

本目录收录指定 Explorer 网页中的全部 **51 个数据集**：13 个回归、30 个二分类、8 个多分类。

来源：[Web Report](https://deqingfu.github.io/tabfm-auto/) · [Pipeline Explorer](https://deqingfu.github.io/tabfm-auto/explorer.html)。

每个数据集目录使用 Explorer 的完整 `name`，大小写、连字符和下划线均保留。论文中的简写可能与此不同，例如 `concrete_strength` 对应网页中的 `concrete_compressive_strength`。

- `pipeline.py`：Code → All code 的完整源码（网页 `code` 字段）。
- `README.md`：Write-up → README.md 的原始 Markdown（网页 `readme` 字段）。

所有原文均以 UTF-8 无损保存，未添加标题、行号或中文注释，未调整格式、辅助函数、导入语句或已有注释。

## 论文与代码的对应关系

已阅读本地 `paper_pdf/TabFM-Auto.pdf` 的正文及相关附录。该工作固定 TabFM 的模型权重，由编码智能体结合数据集元信息与训练集内部三折交叉验证，搜索数据处理流水线。四个主要接口分别是 `preprocess`（清洗与目标变换）、`engineer`（特征工程）、`sample`（上下文行选择）、`postprocess`（预测校准与逆变换）；`TABFM_KWARGS` 用于配置基础模型的推理预设。

论文在每个数据集的 fold 0 训练集上搜索一次，再固定所选代码用于各官方划分。本目录保存网页当前提供的每个数据集的一份最终代码和对应说明。

## 完整性核验

已确认 51 个目录、51 份完整源码、51 份非空 Write-up；102 个文件逐字节匹配网页原始字段，全部 Python 源码通过语法解析。未执行数据集流水线或模型推理。

`manifest.json` 保存每份文件的绝对路径、字节数、SHA-256、原网页定位链接，以及网页快照的 SHA-256 和保存时间。

网页快照与提取脚本保存在工作区 `.tabfm_extract/` 下。可在工作区根目录激活 `tabular` 环境后，运行 `python .tabfm_extract/extract_pipelines.py` 重复核验；已有内容不同时会拒绝覆盖。

## 数据集索引

| 数据集（完整名称） | 类型 | 完整代码 | Write-up |
| --- | --- | --- | --- |
| airfoil_self_noise | 回归 | [pipeline.py](airfoil_self_noise/pipeline.py) | [README.md](airfoil_self_noise/README.md) |
| Amazon_employee_access | 二分类 | [pipeline.py](Amazon_employee_access/pipeline.py) | [README.md](Amazon_employee_access/README.md) |
| anneal | 多分类 | [pipeline.py](anneal/pipeline.py) | [README.md](anneal/README.md) |
| Another-Dataset-on-used-Fiat-500 | 回归 | [pipeline.py](Another-Dataset-on-used-Fiat-500/pipeline.py) | [README.md](Another-Dataset-on-used-Fiat-500/README.md) |
| APSFailure | 二分类 | [pipeline.py](APSFailure/pipeline.py) | [README.md](APSFailure/README.md) |
| bank-marketing | 二分类 | [pipeline.py](bank-marketing/pipeline.py) | [README.md](bank-marketing/README.md) |
| Bank_Customer_Churn | 二分类 | [pipeline.py](Bank_Customer_Churn/pipeline.py) | [README.md](Bank_Customer_Churn/README.md) |
| Bioresponse | 二分类 | [pipeline.py](Bioresponse/pipeline.py) | [README.md](Bioresponse/README.md) |
| blood-transfusion-service-center | 二分类 | [pipeline.py](blood-transfusion-service-center/pipeline.py) | [README.md](blood-transfusion-service-center/README.md) |
| churn | 二分类 | [pipeline.py](churn/pipeline.py) | [README.md](churn/README.md) |
| coil2000_insurance_policies | 二分类 | [pipeline.py](coil2000_insurance_policies/pipeline.py) | [README.md](coil2000_insurance_policies/README.md) |
| concrete_compressive_strength | 回归 | [pipeline.py](concrete_compressive_strength/pipeline.py) | [README.md](concrete_compressive_strength/README.md) |
| credit-g | 二分类 | [pipeline.py](credit-g/pipeline.py) | [README.md](credit-g/README.md) |
| credit_card_clients_default | 二分类 | [pipeline.py](credit_card_clients_default/pipeline.py) | [README.md](credit_card_clients_default/README.md) |
| customer_satisfaction_in_airline | 二分类 | [pipeline.py](customer_satisfaction_in_airline/pipeline.py) | [README.md](customer_satisfaction_in_airline/README.md) |
| diabetes | 二分类 | [pipeline.py](diabetes/pipeline.py) | [README.md](diabetes/README.md) |
| Diabetes130US | 二分类 | [pipeline.py](Diabetes130US/pipeline.py) | [README.md](Diabetes130US/README.md) |
| diamonds | 回归 | [pipeline.py](diamonds/pipeline.py) | [README.md](diamonds/README.md) |
| E-CommereShippingData | 二分类 | [pipeline.py](E-CommereShippingData/pipeline.py) | [README.md](E-CommereShippingData/README.md) |
| Fitness_Club | 二分类 | [pipeline.py](Fitness_Club/pipeline.py) | [README.md](Fitness_Club/README.md) |
| Food_Delivery_Time | 回归 | [pipeline.py](Food_Delivery_Time/pipeline.py) | [README.md](Food_Delivery_Time/README.md) |
| GiveMeSomeCredit | 二分类 | [pipeline.py](GiveMeSomeCredit/pipeline.py) | [README.md](GiveMeSomeCredit/README.md) |
| hazelnut-spread-contaminant-detection | 二分类 | [pipeline.py](hazelnut-spread-contaminant-detection/pipeline.py) | [README.md](hazelnut-spread-contaminant-detection/README.md) |
| healthcare_insurance_expenses | 回归 | [pipeline.py](healthcare_insurance_expenses/pipeline.py) | [README.md](healthcare_insurance_expenses/README.md) |
| heloc | 二分类 | [pipeline.py](heloc/pipeline.py) | [README.md](heloc/README.md) |
| hiva_agnostic | 多分类 | [pipeline.py](hiva_agnostic/pipeline.py) | [README.md](hiva_agnostic/README.md) |
| houses | 回归 | [pipeline.py](houses/pipeline.py) | [README.md](houses/README.md) |
| HR_Analytics_Job_Change_of_Data_Scientists | 二分类 | [pipeline.py](HR_Analytics_Job_Change_of_Data_Scientists/pipeline.py) | [README.md](HR_Analytics_Job_Change_of_Data_Scientists/README.md) |
| in_vehicle_coupon_recommendation | 二分类 | [pipeline.py](in_vehicle_coupon_recommendation/pipeline.py) | [README.md](in_vehicle_coupon_recommendation/README.md) |
| Is-this-a-good-customer | 二分类 | [pipeline.py](Is-this-a-good-customer/pipeline.py) | [README.md](Is-this-a-good-customer/README.md) |
| jm1 | 二分类 | [pipeline.py](jm1/pipeline.py) | [README.md](jm1/README.md) |
| kddcup09_appetency | 二分类 | [pipeline.py](kddcup09_appetency/pipeline.py) | [README.md](kddcup09_appetency/README.md) |
| Marketing_Campaign | 二分类 | [pipeline.py](Marketing_Campaign/pipeline.py) | [README.md](Marketing_Campaign/README.md) |
| maternal_health_risk | 多分类 | [pipeline.py](maternal_health_risk/pipeline.py) | [README.md](maternal_health_risk/README.md) |
| miami_housing | 回归 | [pipeline.py](miami_housing/pipeline.py) | [README.md](miami_housing/README.md) |
| MIC | 多分类 | [pipeline.py](MIC/pipeline.py) | [README.md](MIC/README.md) |
| NATICUSdroid | 二分类 | [pipeline.py](NATICUSdroid/pipeline.py) | [README.md](NATICUSdroid/README.md) |
| online_shoppers_intention | 二分类 | [pipeline.py](online_shoppers_intention/pipeline.py) | [README.md](online_shoppers_intention/README.md) |
| physiochemical_protein | 回归 | [pipeline.py](physiochemical_protein/pipeline.py) | [README.md](physiochemical_protein/README.md) |
| polish_companies_bankruptcy | 二分类 | [pipeline.py](polish_companies_bankruptcy/pipeline.py) | [README.md](polish_companies_bankruptcy/README.md) |
| qsar-biodeg | 二分类 | [pipeline.py](qsar-biodeg/pipeline.py) | [README.md](qsar-biodeg/README.md) |
| QSAR-TID-11 | 回归 | [pipeline.py](QSAR-TID-11/pipeline.py) | [README.md](QSAR-TID-11/README.md) |
| QSAR_fish_toxicity | 回归 | [pipeline.py](QSAR_fish_toxicity/pipeline.py) | [README.md](QSAR_fish_toxicity/README.md) |
| SDSS17 | 多分类 | [pipeline.py](SDSS17/pipeline.py) | [README.md](SDSS17/README.md) |
| seismic-bumps | 二分类 | [pipeline.py](seismic-bumps/pipeline.py) | [README.md](seismic-bumps/README.md) |
| splice | 多分类 | [pipeline.py](splice/pipeline.py) | [README.md](splice/README.md) |
| students_dropout_and_academic_success | 多分类 | [pipeline.py](students_dropout_and_academic_success/pipeline.py) | [README.md](students_dropout_and_academic_success/README.md) |
| superconductivity | 回归 | [pipeline.py](superconductivity/pipeline.py) | [README.md](superconductivity/README.md) |
| taiwanese_bankruptcy_prediction | 二分类 | [pipeline.py](taiwanese_bankruptcy_prediction/pipeline.py) | [README.md](taiwanese_bankruptcy_prediction/README.md) |
| website_phishing | 多分类 | [pipeline.py](website_phishing/pipeline.py) | [README.md](website_phishing/README.md) |
| wine_quality | 回归 | [pipeline.py](wine_quality/pipeline.py) | [README.md](wine_quality/README.md) |
