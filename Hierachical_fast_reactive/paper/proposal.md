# HFRVLA Proposal Draft

> 草稿目的：這份文件先作為 paper 前置的 proposal 說明文件，而不是正式論文初稿。
> 目前先不放實驗數據與 citation，重點是把問題定義、方法定位、目前已完成的實作成果寫清楚。

## 1. 提案摘要

HFRVLA（Hierarchical Fast-Reactive VLA）研究一個核心問題：在不 fine-tune 慢速 VLA planner 的前提下，能不能用一個小型、快速、以 wrist camera 為主要視覺訊號的 residual correction module，修正 frozen SmolVLA action chunk 在執行過程中的誤差。

目前的主線不是重新訓練 SmolVLA，也不是把整個機器人 policy 改成新架構，而是把 frozen `HuggingFaceVLA/smolvla_libero` 當作慢速 planner，保留它產生的 base action chunk，再訓練一個 fast module 對即將執行的 action 做 per-step residual correction：

```text
a_final = a_base + alpha * clip(delta_a)
```

這個設計把研究問題切得很乾淨：如果效果改善，主要原因應該來自 fast residual module 如何利用 wrist-camera evidence 與 frozen slow-planner context 修正 action，而不是來自 slow planner 被重新訓練、改寫或額外加入複雜的 intervention policy。

## 2. 問題定義

Vision-Language-Action policies 通常會一次產生一段 action chunk。這能降低每個 control step 都呼叫大型模型的成本，但也帶來一個執行層面的問題：action chunk 在 rollout 時有一段時間是相對 open-loop 的。當環境狀態、物體位置、接觸狀態或機器人自身動作出現偏差時，base action chunk 可能逐步偏離適合當下的動作。

最直接的解法是更頻繁地重新呼叫 slow VLA，讓 planner 每一步都重新看最新 observation。然而這個做法會增加延遲與運算成本，也不一定適合部署在需要穩定控制頻率的機器人系統中。另一個方向是保留慢速 planner 的 chunk-level planning 能力，但在執行層加入一個輕量、快速的 correction path，讓系統在 chunk 內仍能根據最新的低階視覺與 robot state 做修正。

因此，HFRVLA 的問題可以定義為：

> 給定一個 frozen SmolVLA slow planner 產生的 action chunk，能否訓練一個只更新 fast path 的 wrist-camera residual module，在不改動 slow planner 權重的情況下，提升 chunk execution 的閉環反應能力？

這裡的「wrist-camera residual」需要精確理解：fast module 的 per-step visual correction signal 來自 wrist camera DINO features；但它不是完全 wrist-only policy。fast module 仍然使用 `a_base`、`z_phase`、chunk index、robot state，以及可選的 slow-planner latent context。換句話說，HFRVLA 測的是 wrist visual correction 能否在 frozen VLA context 上修正 action，而不是測一個純 wrist-only policy 能否獨立完成任務。

## 3. 為什麼這個問題重要

這個問題重要，是因為現有 VLA deployment 面臨一個實際 trade-off：大型 VLA 擅長任務語意與長程規劃，但每一步都呼叫大型 planner 的成本高；短 action chunk 或頻繁 replanning 可以提高反應性，但也犧牲推論效率。反過來，長 action chunk 可以降低 planner 呼叫頻率，卻會放大 execution drift。

HFRVLA 嘗試在兩者中間建立一個分層控制形式：

- slow system 負責 task-level planning 與 action chunk proposal；
- fast system 負責用 wrist evidence 對即將送出的 action 做局部修正；
- slow planner 保持 frozen，讓研究焦點集中在 correction module；
- fast path 小而可重訓，讓資料、訓練與部署成本可控。

如果這個方向成立，它提供的是一個 VLA deployment pattern：不必每次都重訓或 fine-tune 大型 VLA，也不必每個 control step 都呼叫 slow planner，而是用一個小型 correction module 補足 chunk execution 的反應性。

## 4. 目前方法定位

目前 paper 主線採用 A2C2-Wrist 風格的簡化 residual correction，而不是早期 proposal 裡的 gated HFRVLA 設計。這個決定讓第一版 paper 的技術主張更容易解釋，也更貼近目前已完成的實作。

目前主線包含：

- slow planner：frozen `HuggingFaceVLA/smolvla_libero`；
- fast visual signal：wrist camera 經 DINOv3 patch features；
- fast context：robot state、`a_base`、chunk position、`z_phase`，以及 previous/current frame conditioning；
- output：full 7D `delta_a` residual；
- training target：expert action 與 frozen base action 的 raw residual；
- inference merge：`a_base + alpha * clip(delta_a)`；
- current mainline 不使用 learned gate、contact auxiliary loss、gate-adjusted merge，也不把 GRU 作為主要論文敘事。

這個定位的核心優點是解釋乾淨：我們不把改善歸因於一個學到何時 intervention 的 gate，也不把結果混入 contact auxiliary signal。第一版研究先回答更基本的問題：只用 wrist-camera residual correction，加上 frozen slow-planner context，是否已經能修正 action chunk execution 中的部分錯誤？

## 5. 方法概念

HFRVLA 的系統可分成兩層。

第一層是 frozen SmolVLA slow planner。它負責根據 LIBERO observation、language/task input 與 robot state 產生 action chunk。HFRVLA 不更新這個模型的權重；訓練時也避免在 fast module training loop 裡重跑 SmolVLA。

第二層是 trainable fast residual module。它讀取已快取的 wrist DINO features、robot state、base action、chunk-relative information，以及 slow planner 中抽出的 latent context。fast module 預測當前 step 的 `delta_a`，再由 inference-time merge rule 把 residual 加回 `a_base`。

訓練資料不是 raw LIBERO demo 直接餵進 policy。HFRVLA 先建立一個自訂 LeRobotDataset v3，把 frozen SmolVLA 的 base action、slow-planner hidden features、wrist DINO patches、chunk index 與 expert action 都記錄下來。這讓 training loop 可以成為一個 offline residual learning problem：

```text
target_delta = action_expert - a_base
delta_pred   = FastModule(wrist_features, state, a_base, chunk_context)
loss         = MSE(delta_pred, target_delta)
```

這個資料設計很重要，因為它把 slow planner provenance 固定住。若未來更換 slow planner，就必須重新記錄 dataset，否則 `a_base`、`z_goal`、`z_phase` 與 fast residual target 會不一致。

## 6. 目前已完成的實作成果

目前已完成的工作主要是把 HFRVLA 從概念推到 LeRobot-native 的可訓練、可評估 artifact。

已完成的實作包括：

- 建立 `lerobot_policy_hfrvla` LeRobot policy plugin，並以 entry point 註冊 `--policy.type=hfrvla`；
- 實作 `HFRVLAConfig` 與 `HFRVLAPolicy`，讓 policy 可以包住 frozen SmolVLA 並只訓練 fast module；
- 保留 frozen SmolVLA slow planner 的 action chunk 作為 `a_base`，並支援 fast residual merge；
- 建立 Fast Wrist Residual / A2C2-Wrist 主線，使用 wrist DINO patches、state、base action、chunk index、slow-planner context 與 previous correction 預測 full 7D residual；
- 建立 chunk-aware FWR-v2 path，讓 fast module 可以讀取 full frozen base action chunk 與 current chunk step index；
- 建立 HFRVLA custom LeRobotDataset v3 recording pipeline，把 SmolVLA outputs、DINOv3 wrist patches、action residual target 與相關 metadata baked into dataset；
- 建立 frame-level fast-cache pipeline，避免 training loop 重複讀 raw images 或重跑 slow planner / DINO；
- 將 `seq_len` 明確移到 training-time loader/windowing，而不是 cache construction；
- 建立 `scripts/train_via_lerobot.py` wrapper，讓 training 仍走 LeRobot-native workflow，同時支援 HFRVLA 的 fast-cache dataset；
- 建立 checkpoint packaging script，把 fast weights 與 frozen SmolVLA 組成可供 `lerobot-eval` 使用的 policy artifact；
- 建立 eval registry，將 action-step sweep、matched chunk sweep、alpha/clip sweep、FWR sweep 等結果集中管理。

這些成果代表目前 HFRVLA 已經不只是架構草圖，而是有完整資料合成、訓練、封裝、評估與結果登錄流程的研究系統。這份 proposal 後續要補的不是「能不能跑起來」，而是把問題敘事、實驗比較與 paper claim 收斂到一個清楚的版本。

## 7. 預期論文主張

目前最穩健的 paper claim 應該保持在下列範圍內：

1. **Frozen VLA action chunks 可以被小型 wrist-camera residual module 修正。**
   HFRVLA 不 fine-tune slow planner，而是檢驗 execution-time correction 是否能改善 frozen action chunk。

2. **簡化 residual merge 是第一版 paper 的主要敘事。**
   目前主線使用 `a_base + alpha * clip(delta_a)`，不把 learned gate 或 contact auxiliary loss 作為主要貢獻。

3. **HFRVLA 建立了一個 LeRobot-native 的可重現 artifact。**
   方法不只是 isolated model head，而是包含 dataset recording、feature caching、training wrapper、checkpoint packaging 與 eval registry 的完整 pipeline。

4. **planning horizon 與 execution/replan interval 必須分開討論。**
   這是 HFRVLA paper 的重要評估觀點，因為 default every-step replanning 不是長 action-chunk execution 的 matched baseline。

## 8. 本提案暫不主張的內容

為了避免 proposal 過度承諾，以下內容目前不作為第一版主張：

- 不主張已完成 real-hardware validation；
- 不主張 learned gate 或 contact auxiliary loss 是目前主線貢獻；
- 不主張整個 policy 是 wrist-only；
- 不主張 HFRVLA 已完全解決 long-chunk open-loop drift；
- 不主張可以任意更換 slow planner 而不重建 dataset；
- 不主張目前結果能直接外推到所有 VLA backbone 或所有 robot platforms。

這些項目可以作為 future work 或後續 paper extension，但不應該混入目前 proposal 的核心 claim。

## 9. 後續需要補齊的證據

這份 proposal 下一步應補三類證據。

第一，補完整 related work 與 citation verification。尤其要確認 residual-on-frozen-VLA、A2C2-style correction、fast-slow robot control、wrist-camera manipulation、LeRobot / SmolVLA 相關 work 的準確引用。所有 citation 必須經過查證後再放進正式 paper。

第二，補定量實驗段落。這份草稿刻意不放數據，但正式 paper 需要把 eval registry 中的 sweep 結果轉成清楚的表格與圖，並區分 action-step sweep、matched chunk sweep、alpha/clip calibration、FWR-v2 sweep。

第三，補 error analysis。HFRVLA 的重點不只是成功率是否上升，而是理解 residual correction 在什麼 execution setting 有效、什麼 setting 失效，以及失效來自 base-policy drift、residual overcorrection、wrist observability limit，還是 training/deployment mismatch。

## 10. Grill-Me 決策檢查

以下是目前最關鍵的設計問題與推薦答案，作為後續討論起點。

**問題 1：第一版 proposal 應該以 gated HFRVLA 還是 A2C2-Wrist mainline 為主？**
推薦答案：以 A2C2-Wrist / Fast Wrist Residual mainline 為主。因為這是目前較新的實作與 paper direction，也避免把 gate/contact auxiliary 這些未成為主線的設計寫成核心 claim。

**問題 2：是否要把「wrist-only」寫成主要 novelty？**
推薦答案：可以，但要精確寫成「per-step visual correction signal is wrist-camera based」。不要寫成整個 policy wrist-only，因為 fast module 還使用 `a_base`、`z_phase`、state 與 chunk context。

**問題 3：是否要在 proposal 內放目前數據？**
推薦答案：不要。這份文件先定義問題與成果，正式 paper draft 再從 eval registry 匯入數據、圖與表格。

**問題 4：目前最大的 paper 風險是什麼？**
推薦答案：不是系統還沒實作，而是 claim 容易被寫得過大。第一版應聚焦在 frozen SmolVLA action chunk 的 wrist residual correction，而不是承諾 real hardware、universal VLA generality 或 learned gate semantics。

## 11. 建議的一句話版本

HFRVLA proposes a LeRobot-native fast wrist residual module that corrects frozen SmolVLA action chunks at execution time, testing whether lightweight wrist-camera correction can improve VLA chunk execution without fine-tuning the slow planner.

中文版本：

> HFRVLA 提出一個 LeRobot-native 的 wrist-camera fast residual module，在不 fine-tune frozen SmolVLA slow planner 的前提下，對 action chunk 執行過程中的 base action 做 per-step residual correction，檢驗小型 fast module 是否能補足 VLA chunk execution 的閉環反應能力。
