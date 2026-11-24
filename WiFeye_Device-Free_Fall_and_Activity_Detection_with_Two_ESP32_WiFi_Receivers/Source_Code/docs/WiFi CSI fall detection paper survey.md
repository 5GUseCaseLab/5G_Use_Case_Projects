# Fix the CSI pipeline before the models

SafeSense-5G should fix its data layer first. After that it should add per-user calibration, temporal structure and event-level fall logic. DANN and synthetic falls belong in controlled ablations, not in the default pipeline. Our measurements (2026-11-20) explain why. Every raw capture interleaves CSI from about eight transmitters. Invalid subcarriers entered the features. A ~2 Hz high-pass discarded the posture change that best separates falls, sits and stands. As a result the original pipeline scored a **fall-vs-ADL AUC of 0.48 under leave-one-subject-out (LOSO), which is chance**. Filtering to the dominant source, dropping subcarriers 0 and 27-37 and using 20 posture/burst features lifted it to **0.76, with 46.8% fall sensitivity at 8.0% ADL false alarms**, in one unreplicated run. That is a larger shift than any model or augmentation change we tried. The 2014-2026 literature ranks the next levers clearly:

- **One to five labelled trials per class from a new user** recover most of a 40-60 point cross-subject drop. ESP32 falls go from 59.9% zero-shot to 77.6-84.0% at one shot.
- **An activity state machine** added 20 points on sit/stand transitions.
- **Event pipelines with post-fall confirmation** hold the lowest false-alarm rates.

By contrast, **DANN ties plain training** with four source domains and label shift. **Augmentation adds 0-3 points** when the test person is unseen, which matches our null augmentation result (AUC 0.45-0.53). A second ESP32 should add about 2-5 points. Our single run showed RX1-only AUC 0.704 against 0.762 fused: the right size, but untested. Subject 3's missing falls should stay missing. Report that fold as false alarms only, test any "subject-style" augmentation with a leave-one-cell-out simulation, consider fall models that need no falls from the user, and record real falls. The project can credibly contribute five things: a documented ESP32 source-contamination fix, irregular-sampling statistics, a controlled 1-vs-2-receiver LOSO fall ablation, a missing-cell protocol and a measured tier cascade. It cannot claim person independence, cross-environment robustness, the 95%/3% targets or 5G MEC performance.

## Source mixing, not model choice, produced the chance-level baseline

Our measurements show that the original pipeline was largely classifying contamination. **Each raw file mixes CSI from several transmitters.**

- About 80% of packets come from one HT (sig_mode=1) source whose logged MAC equals the receiver's own MAC.
- The rest come from about seven other MACs. These include about 10% legacy (sig_mode=0) frames and a probable neighbouring AP.
- Consecutive-packet amplitude jumps are **4× larger across sources (8.3) than within one source (2.1)**, so every source switch looks like motion.
- As a result, empty-room trials carry roughly half the motion energy of human trials.

Two more defects compounded this. Subcarrier index 0 (value ~107, the highest variance) is not channel data, and indices 27-37 are null, yet all of them entered the features. The 0.5 s moving-average subtraction, a ~2 Hz high-pass, then removed slow posture change, even though the start-vs-end static level shift relative to noise is the clearest cue in the data:

| Classes | Start-vs-end level shift / noise |
|---|---|
| Fall, sit, stand | 0.6-0.9 |
| Hand, walk | 0.3-0.5 |
| Empty room | 0.2-0.3 |

The resulting trial-level LOSO scores (our measurements) were near chance:

| Model / protocol | Accuracy | Macro-F1 | Fall-vs-ADL AUC | Fall sensitivity | ADL false alarms |
|---|---|---|---|---|---|
| Random forest, LOSO | 33.6% | 0.30 | 0.48 | **3.8%** | 9.2% |
| HistGradientBoosting, LOSO | 31.0% | — | — | 57% | 49% |
| Random forest, trial-grouped 5-fold | 48.7% | — | — | — | — |

HistGradientBoosting sat at a different point on the same chance curve. Even trial-grouped 5-fold only reached **48.7%, so seen subjects failed too**. The features, not only the person shift, were the problem.

The proof of concept isolates the cause. It applied single-source filtering, dropped the bad subcarriers and used 20 simple trial-level posture/burst features with a random forest under the same LOSO. It reached:

- accuracy 44.2%, macro-F1 0.444, and fall-vs-ADL AUC 0.762;
- **46.8% fall sensitivity (37/79; Wilson 95% interval about 36-58%, our calculation) at 8.0% ADL false alarms**;
- with RX1 only, accuracy 40.3% and AUC 0.704.

This was a single run with no significance test (our measurements). At an 8% false-alarm rate a chance detector would catch 8% of falls; this one catches about six times that. The remaining confusions are sit↔stand and hand↔empty_room. The v2 pipeline now under construction formalizes the same fixes: a dominant-source filter, the subcarrier mask, median (posture-preserving) static removal, scale-free per-RX posture/motion features, and label-free event-centred labels.

The literature corroborates each defect, even though **none of the surveyed ESP32 papers reports which transmitters produced its CSI**.

- **Subcarrier mask.** Dropping index 0 and indices 27-37 leaves exactly 52 subcarriers. That matches the L-LTF data subcarriers kept in ESP32-S3 studies ([Strohmayer & Kampel](https://arxiv.org/html/2401.00964v1)), the WiFall dataset ([KNN-MMD](https://arxiv.org/html/2412.04783v5)) and the router-TX/ESP32-RX dataset of Lim & Leow ([Data in Brief 2024](https://doi.org/10.1016/j.dib.2024.111101)).
- **Index 0.** ESP-IDF v4.3's CSI structure carries a `first_word_invalid` flag ([v4.3 header](https://raw.githubusercontent.com/espressif/esp-idf/v4.3/components/esp_wifi/include/esp_wifi_types.h)). It is our hypothesis, not a finding, that index 0's anomaly is that artefact; logging the flag would settle it.
- **Mixed packet types.** FuseFi built a sanitization stage specifically to harmonize heterogeneous packet types and remove burst redundancy. It notes that single-frame-type methods discard about 70% of packets ([FuseFi](https://arxiv.org/html/2512.22143)).
- **ESP32 data quality.** CSI-Bench puts "ESP32-based low-quality CSI" in its hard test tier ([CSI-Bench](https://arxiv.org/html/2505.21866v2)).
- **Traffic source.** Espressif recommends a dedicated transmitter as the most reliable traffic mode ([esp-csi](https://github.com/espressif/esp-csi)).
- **Packet loss threshold.** Origin's two-year home deployment admits devices only below 10% packet loss ([Zhu et al. 2025](https://arxiv.org/html/2506.04322v1)).

The "own MAC" observation stays unexplained. ESP-IDF v4.3 exposes only a `mac` field and no 802.11 header. v5.2 adds `hdr`, and v5.5 adds `rx_seq` ([v5.2](https://raw.githubusercontent.com/espressif/esp-idf/v5.2/components/esp_wifi/include/esp_wifi_types.h); [v5.5](https://raw.githubusercontent.com/espressif/esp-idf/v5.5/components/esp_wifi/include/local/esp_wifi_types_native.h)). The frame type behind the dominant stream therefore cannot be read from the current logs.

The packet timing is set by the serial link, not the radio. The measured median gap of 7 ms (~140/s) matches a back-of-envelope estimate from the notes, which is an inference rather than a sourced figure:

- An ASCII `CSI_DATA` line is about 600-700 bytes.
- At 921600 baud (~92 kB/s), that caps output at roughly 130-150 lines/s.

Three sources support this:

- The ESP32 can produce more than 1000 CSI/s when nothing is sent over serial ([Hernandez & Bulut, COMST](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf)).
- The tool's README stresses that "the higher baud rate the better" ([ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)).
- At 230,400 baud, ASCII output capped near 30 packets/s and binary near 85 ([Armenta-Garcia et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526573/)).

About a fifth of SafeSense-5G's serial budget carries packets that are later discarded (our inference).

Arrival is also bursty. About 18-20% of gaps exceed 20 ms, the typical maximum gap is about 125 ms per trial, and one trial has a 1 s hole. The median rate is about 70 Hz, which **caps usable bandwidth near 35 Hz and lower inside gaps**. Even ping-driven 100 Hz ESP32 collection loses 14.51% of packets on average, and 1-70% within single seconds ([CSI-BERT](https://arxiv.org/html/2403.12400v1)). The consequence is strategic: on this hardware, the robust cues are low-frequency ones such as posture level, burst envelope and stillness. That is exactly what the proof of concept used.

Two protocol confounds also need control.

- **Session blocks.** Activities were recorded in consecutive 20-trial blocks per subject. Empty room was recorded on a separate day with about 5 dB higher RSSI (our measurements). The evidence on drift conflicts:
  - an ESP32 HAR model lost **52% relative accuracy when tested 10 minutes after training** ([Matey-Sanz et al., IPIN 2023](https://doi.org/10.1109/IPIN57070.2023.10332214), abstract only);
  - a Nexmon home-health study found temporal drift minimal when room and people stayed constant ([Monjur et al.](https://arxiv.org/html/2402.09452v1)).
  
  Separately, ESP32 sit/stand CSI identifies 15 participants with 84.29% and 74.13% accuracy ([Custance et al. 2023](https://doi.org/10.3390/electronics12214500), abstract only). Session and identity signatures are therefore strong enough to be learned instead of activity, and the project must measure them rather than assume them away.
- **Script timing.** Subjects did not follow the 2-6 s script.
  - The fall motion peak has a median of 2.6 s; 25% of peaks come before 2.1 s and 10% before 1.1 s.
  - Per-subject medians are 2.5 s for subjects 1 and 2, 3.95 s for subject 4 and 5.2 s for subject 5.
  
  Fixed-interval labels would therefore mislabel windows and also turn each subject's timing into a feature. Label-free event-centred segments remove both problems.

### What this survey corrects in the earlier report

The earlier report is superseded on the points below. Where the new notes and our measurements disagree with it, they win.

| Earlier report said | Correction | Basis |
|---|---|---|
| Realistic prior: trial-grouped CV in the mid-to-high 90s, LOSO 75-88% | Measured trial-grouped 48.7% and LOSO 33.6% (original pipeline), LOSO 44.2% (proof of concept). The prior holds only for clean, single-source, regularly sampled CSI. Ye et al.'s 5-subject ESP32 study drove its traffic with STA ping and AP reply at 100 packets/s | Our measurements; [Ye et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591964) |
| The main fix is longer windows ("lengthen windows, not models") | The main fix is the data layer: source filtering, a 52-subcarrier mask and posture-preserving static removal moved AUC from 0.48 to 0.76. Window and segment length still matter, but second | Our measurements |
| Label fall windows from the scripted 2-6 s interval | The script was not followed: 10% of fall peaks come before 1.1 s, and subject 5's median is 5.2 s. Use label-free, event-centred labels | Our measurements |
| Normalize amplitude by the median of the 0-2 s still segment | The first 2 s are not reliably still. Use the trial median or detected still segments, and normalize against a quiet baseline rather than z-scoring each stream. Plain z-scoring cut ESP32 accuracy from 87.36% to 70.39% in one test | Our measurements; [Hernandez & Bulut](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf) |
| Add 10-20 and 20-40 Hz band-energy features (fall Doppler about 16-48 Hz) | With a median rate of about 70 Hz and 18-20% of gaps over 20 ms, usable bandwidth is at most about 35 Hz, and lower inside gaps. Defer high-band features until the TX rate is fixed, and mask gaps rather than interpolate across them | Our measurements; [FuseFi](https://arxiv.org/html/2512.22143) |
| ESP32 streams arrive at 50-130 Hz | The median gap of 7 ms is set by ASCII serial output at 921600 baud. The median rate is about 70 Hz, with bursts and a 1 s hole | Our measurements; [Armenta-Garcia et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526573/) |
| Adversarial domain adaptation is out of scope, and EI's numbers were unavailable | EI reached 0.75 only with 22 source domains. On ESP32 person domains, DANN added 14 points on falls zero-shot but stayed below one labelled shot, and DomainBed finds ERM ≈ DANN. Because the project documents name a DANN server-tier classifier, run it as a λ=0-controlled ablation | [EI](https://cse.buffalo.edu/~lusu/papers/MobiCom2018.pdf); [KNN-MMD](https://arxiv.org/html/2412.04783v5); [DomainBed](https://ar5iv.labs.arxiv.org/html/2007.01434) |
| "DapFall/FallDar": 5.7% FAR / 3.4% miss and 95.28-99.67% | These are two papers. FallDar (IEEE TMC 2022) reports 5.7% FAR and 3.4% missed alarms. AmFall (IEEE IoT-J 2025) reports 95.28-99.67%. Both figures are from abstracts only | [FallDar](https://doi.org/10.1109/TMC.2022.3188779); [AmFall](https://doi.org/10.1109/JIOT.2025.3585401) |
| DeFall used 5.8 GHz / 40 MHz hardware | The full text gives an Intel 5300 with 3 antennas and 30 subcarriers on 5 GHz channel 153. Its fall template came from a dummy, not from the test users | [DeFall PDF](https://originwirelessai.com/wp-content/uploads/2024/04/DeFall.pdf) |
| "Zhou et al." (TIoT) reached about 84% leave-one-person-out (snippet) | The first author is Ma. The full text gives 84% for a plain separable CNN and 97% with an activity-transition state machine | [Ma et al.](https://gzhou.pages.wm.edu/wp-content/blogs.dir/5736/files/sites/13/2017/12/ACM_TIOT_2020.pdf) |
| Mild ±10-20% time-warping is a plausible fall augmentation | Time warping significantly hurt MLP and LSTM-FCN models across 128 datasets, and it rescales descent speed. Cap it at ±10%, or replace it with window warping or STFT-window diversity | [Iwana & Uchida](https://pmc.ncbi.nlm.nih.gov/articles/PMC8282049); [RFBoost](https://arxiv.org/html/2410.07230v1) |

## Calibration and temporal structure beat adversarial alignment for new users

The table below summarizes which methods fit each SafeSense-5G problem. The sections that follow give the evidence.

| Problem | Use now (strongest evidence) | Test as a controlled ablation | Deprioritize at 509 trials |
|---|---|---|---|
| Person independence | v2 data layer (ours: AUC 0.48→0.76). k-shot calibration: ESP32 falls 59.9%→77.6-84.0% at 1 shot ([KNN-MMD](https://arxiv.org/html/2412.04783v5)); ESP32 gestures 64.8%→84.5% ([CrossFi](https://arxiv.org/html/2408.10919v4)); 41.6%→91.3% with 5 windows per class ([FuseFi](https://arxiv.org/html/2512.22143)) | Activity state machine or HMM: 77%→97% leave-one-person-out ([Ma et al.](https://gzhou.pages.wm.edu/wp-content/blogs.dir/5736/files/sites/13/2017/12/ACM_TIOT_2020.pdf)). Quiet-baseline per-RX normalization ([OpenCSI](https://doi.org/10.1109/LCN67947.2026.11660791), abstract only). Subject-ID probe, CORAL, class-conditional DANN | Plain DANN: 66.4 vs ERM 67.0 ([DomainBed](https://ar5iv.labs.arxiv.org/html/2007.01434)). MAE or contrastive pre-training: −5.5 points at about 1.3k samples ([Jiang et al.](https://arxiv.org/html/2511.18792v1)). MAML: the strongest claim leaks target subjects ([TCN-MAML](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252493/)) |
| Fall detection | Event-centred trigger → segment → classify → stillness confirmation ([SiFall](https://arxiv.org/html/2301.03773v2); [LT-Fall](https://doi.org/10.1145/3580835)). Threshold moving inside inner folds ([CDL-Fall](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512051/)). In-fold tier recalibration with a post-fall latch | Speed/burst template with DTW ([DeFall](https://originwirelessai.com/wp-content/uploads/2024/04/DeFall.pdf)). One-class model trained on ADLs only, with no falls ([SiFall](https://arxiv.org/html/2301.03773v2); [mmFall](https://doi.org/10.1109/TASE.2020.3042158)) | Deep nets on raw short windows: 59.91% on ESP32 without augmentation ([Armenta-Garcia](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526573/)). SMOTE, focal loss |
| 1 vs 2 ESP32 | Late or shared-weight fusion after per-RX normalization. SHARP: late fusion 96.59% vs early 81.68% ([SHARP](https://ar5iv.arxiv.org/html/2103.09924)). Gain alignment ([IoT-J 2026](https://doi.org/10.1109/JIOT.2026.3667804)) | OR/AND fall-decision fusion; per-class complementarity | 128-channel early concatenation as the default ([dilution study](https://arxiv.org/html/2602.10823v3)) |
| Subject 3's missing falls | Report the fold as false alarms only, pool fall recall over 4 subjects, use class weights, record real falls | Leave-one-cell-out test of style augmentation; fall models that need no falls from the user | Duplicating or relabelling falls. Generators trained on 79 falls: unfiltered synthetic CSI cost −13.4% ([Gong et al.](https://arxiv.org/abs/2506.23174)) |

### The size of the cross-subject drop

The new cross-subject papers agree on the size of the problem. Without any data from the target user, learned models on small cohorts lose about 40-60 points:

- **CrossGR:** a CNN fell from 95% to **36% on unseen users** ([CrossGR](https://eprints.whiterose.ac.uk/170933/6/ubicomp2021.pdf)).
- **CrossFi:** a ResNet-18 on ESP32-S3 gestures fell from 80.75% to **40.84%** ([CrossFi](https://arxiv.org/html/2408.10919v4)).
- **CSI-Bench:** a Transformer fell from 88.06% to **54.72%** cross-user ([CSI-Bench](https://arxiv.org/html/2505.21866v2)).

The results that hold above 90% rest on physics features. SHARP, trained on one person, scored 99.33% on a new person and 95.99% in a new room. DeFall's dummy-built fall template detected 91.67-100% of falls for each of 10 real users ([SHARP](https://ar5iv.arxiv.org/html/2103.09924); [DeFall](https://originwirelessai.com/wp-content/uploads/2024/04/DeFall.pdf)). Both need hardware this project lacks:

- SHARP sanitizes phase per antenna on 80 MHz Nexmon with 4 RX antennas.
- DeFall estimates speed on a 3-antenna Intel 5300 at 5 GHz.

No paper has computed such features from single-antenna 20 MHz ESP32 CSI under LOSO.

Generic statistics fed to RF, SVM or KNN collapse on unseen users. In CrossGR, t-SNE of five users' CSI formed five clusters **by user, not by gesture**. CrossGR's failing feature list (min, max, variance, skewness, kurtosis, FFT peaks) closely matches the original 22-feature pipeline ([CrossGR](https://eprints.whiterose.ac.uk/170933/6/ubicomp2021.pdf)).

### Few-shot calibration is the most consistent non-adversarial fix

Calibrating on a few labelled samples from the new user is the most reliable lever:

| Study | Zero-shot | With calibration |
|---|---|---|
| CrossFi (gestures) | 64.81% | 84.47% at one shot; 91.72% with MK-MMD; near in-domain at about 5 shots per class ([CrossFi](https://arxiv.org/html/2408.10919v4)) |
| KNN-MMD (ESP32-S3, binary falls, 10 people) | 59.86% | 77.62-84.03% at one shot ([KNN-MMD](https://arxiv.org/html/2412.04783v5)) |
| WiGr (prototypical, cross-environment) | — | 60 / 65 / 75 / 83% at 1 / 2 / 3 / 4 shots ([WiGr](https://zhang-xie.github.io/papers/WiGr_zhang.pdf)) |
| FuseFi (refit linear head only) | — | 91.3% from 5 labelled windows per activity ([FuseFi](https://arxiv.org/html/2512.22143)) |

Two cautions apply to the KNN-MMD numbers. Its few-shot baselines report the best test accuracy reached during training, which is optimistic. Plain kNN on ESP32 falls scored only 49.63% at one shot.

For a tree pipeline, the practical analogue is to add k = 1-3 ten-second trials per class from the held-out subject. Then compare a random forest refit with those trials up-weighted against a nearest-class-mean classifier on standardized features.

Calibration also targets the remaining sit↔stand confusion. The sign and size of a posture's static CSI level depend on where that person stands relative to each link. Which static state means "seated" therefore does not transfer across people, but one labelled trial from that person reveals it. This is our inference, not a published result.

### Temporal structure and pre-training

Temporal structure is the second cheap lever. Ma et al.'s rule-based activity-transition state machine, applied over per-window CNN outputs, lifted leave-one-person-out accuracy from 77% to 97% on five activities that include sit-down and stand-up. On FallDeFi data with unseen persons and rooms, it went from 64% to 83% ([Ma et al.](https://gzhou.pages.wm.edu/wp-content/blogs.dir/5736/files/sites/13/2017/12/ACM_TIOT_2020.pdf)).

Self-supervised and foundation models do not fit this data scale:

- **Jiang et al.** (14 corpora, none of them ESP32): MAE fine-tuning raised Widar3.0 cross-subject accuracy from 54.7% to 67.5%. But pre-training on about 1,300 samples scored 5.5 points *below* supervised training, and parity arrived only near 13,200 samples ([Jiang et al.](https://arxiv.org/html/2511.18792v1)).
- **Contrastive methods:** under linear evaluation they trail supervised learning by 10-50 points in-domain ([Xu et al.](https://ar5iv.labs.arxiv.org/html/2308.02412)).
- **AutoFi** scored 48.05% on ESP32 one-shot new-person falls ([KNN-MMD](https://arxiv.org/html/2412.04783v5)).

### DANN should earn its place in the server tier

The project documents name a DANN classifier for the server tier. The evidence says it should have to earn that place.

- **Shallow networks on fixed features.** In Ganin et al.'s own experiments, a one-hidden-layer network on fixed features gained only about **+0.2 to +1.0 point** from the domain head, and lost on some pairs ([Ganin et al., JMLR 2016](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf)).
- **Few source domains.** DomainBed, with 3-6 domains per dataset, found ERM 67.0, CORAL 67.2, MMD 66.8 and DANN 66.4 ([DomainBed](https://ar5iv.labs.arxiv.org/html/2007.01434)).
- **Same backbone, no augmentation.** On Widar3.0, domain-adversarial training without augmentation moved F1 from 40.62 to 42.02, within one standard deviation. Augmentation alone reached 49.32, and the two combined 65.66 ([DATTA](https://arxiv.org/html/2411.13284)).
- **Many domains help.** EI needed 22 subject-room source domains to reach 0.75, and called accuracy "low" with only 2 source domains ([EI](https://cse.buffalo.edu/~lusu/papers/MobiCom2018.pdf)).
- **Hardware-matched case.** On ESP32 person domains, DANN raised binary fall accuracy from 59.86% to 74.06%. One labelled shot still beat it, at 77.62-80.93% ([KNN-MMD](https://arxiv.org/html/2412.04783v5)).

SafeSense-5G also has a specific trap: label shift. Invariant representations carry an information-theoretic error lower bound when label distributions differ across domains ([Zhao et al., ICML 2019](https://proceedings.mlr.press/v97/zhao19a.html)). Here, subject 3 has no falls and the empty-room class has no subject. An unconditional discriminator can therefore identify domains from "has falls", and gradient reversal would push fall features toward subject 3's non-fall features.

A fair DANN test needs the following:

- **Class-conditional alignment**, or exclusion of fall and empty-room windows from the domain loss.
- **The same MLP with λ = 0** as the decisive control.
- **Nested source-subject selection** of λ and early stopping, or reverse validation.
- **5-10 seeds.**
- **Four separately named settings:**
  - domain generalization;
  - inductive UDA, with a trial-grouped split of the held-out subject as in CsiGAN ([CsiGAN](https://www.github.com/ChunjingXiao/CsiGAN));
  - transductive UDA, as in EI;
  - k-shot calibration.

DomainBed's search-snippet figure, about 7 points of inflation from test-domain oracle model selection, is **unverified** ([arXiv 2007.01434](https://arxiv.org/pdf/2007.01434)). No Wi-Fi paper applies DANN to handcrafted feature vectors, so the ablation is novel even if it comes out null.

Normalization deserves the same care. Per-link z-scoring against a quiet-period baseline gave occupancy F1 up to 0.99 zero-shot, against 0.87 for per-session normalization ([OpenCSI, LCN 2026](https://doi.org/10.1109/LCN67947.2026.11660791), abstract only). Plain z-score standardization, by contrast, cut ESP32 accuracy from 87.36% to 70.39% ([Hernandez & Bulut](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf)). The lesson matches the posture finding: **normalize against an empty or still reference, never against the trial's own variance**. Any per-subject normalization that uses the held-out subject's unlabelled windows is transductive and must be reported separately.

## Fall detection rewards event pipelines, honest negatives and calibrated gates

### Event pipelines hold the lowest false-alarm rates

The lowest false-alarm systems all detect an event, segment it, and then confirm it. SiFall is the best-documented example:

- **Pipeline:** an acceleration gate above 2.5 m/s² triggers an event segment (average 5.1 s). An autoencoder trained only on normal activities then flags anomalies.
- **Trained setting:** **100% TPR at 1.8% FPR over 19.3 h**, against 64.9% TPR and 49.2% FPR for RT-Fall on the same fall-like set.
- **New testbed:** accuracy fell to 78.3% without fine-tuning.
- **Alarm delay:** 1.67 s.
- Source: [SiFall, SenSys 2022](https://arxiv.org/html/2301.03773v2).

Other systems follow the same pattern. All of the following figures come from abstracts only:

| System | Sensor | Confirmation step | Reported result |
|---|---|---|---|
| LT-Fall | mmWave | Monitors post-fall micro-motion (long-lie) | Zero false alarms ([LT-Fall](https://doi.org/10.1145/3580835)) |
| mmFall | mmWave | Requires an anomaly spike and a centroid-height drop together | 98% of 50 falls, 2 false alarms ([mmFall](https://doi.org/10.1109/TASE.2020.3042158)) |
| FallAware | Wi-Fi | Physics features plus a 21k-parameter LSTM | 94.1% in unseen environments; under 5 false alarms per month in homes ([FallAware](https://doi.org/10.1109/JSAS.2024.3520517)) |
| FallDar | Wi-Fi | Body-speed features | 5.7% FAR, 3.4% missed alarms over six months ([FallDar](https://doi.org/10.1109/TMC.2022.3188779)) |

DeFall shows a fall detector does not need the user's falls at all: its fall template came only from 846 dummy falls ([DeFall](https://originwirelessai.com/wp-content/uploads/2024/04/DeFall.pdf)). Plain CNN or LSTM classifiers report 90-99% only on same-domain or random splits.

### What SafeSense-5G's tiers and labels need

Three of our measurements apply directly here.

- **Event timing.** The timeline was not followed, so segments must be centred on the detected motion peak. A fixed script interval will not work.
- **Uncalibrated tier thresholds.** With dormant at 1.5 and fall fast-path at 15.0, **19,325 of 19,333 windows (99.96%) route to the fall fast-path**. That is consistent with cross-source amplitude jumps inflating the energy-derivative cue (our inference). The thresholds must be refit after source filtering, from training-fold ADL distributions. Report the fraction of windows exiting at each tier and the falls lost at each gate.
- **A post-fall latch.** A motionless occupant is hard to tell from an empty room with variance-style features ([CRONOS](https://arxiv.org/pdf/2211.10354), snippet only). Tier 1 must therefore latch after a Tier-2 fall trigger, not drop back to "dormant" while the person lies still (our inference).

TCS-Fall's stream-level metrics map directly onto 10 s trials ([TCS-Fall](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11151769/fullTextXML)):

- **"Correct warning"** is per-trial sensitivity.
- **"Error warning"** is a per-trial false alarm.

### The negative set is the easy one

SafeSense-5G's negatives (walk, sit, stand, hand, empty room) are the "easy" set, and similar-class studies report 95-98% ([Taha et al.](https://doi.org/10.1109/SENSORS47087.2021.9639715); [Damodaran et al.](https://doi.org/10.1007/s42486-020-00027-1)). The documented confusers are absent:

- sit-to-floor, the main confuser in [SiFall](https://arxiv.org/html/2301.03773v2);
- squat, in [TED-Net](https://arxiv.org/html/2504.16655);
- sit/stand (86.7%) and pick-up (90%), the weakest live classes in the 2026 physics-driven transformer ([Wang et al.](https://arxiv.org/html/2605.00869)).

Chu et al. note that systems trained on more comprehensive datasets report *lower* accuracy ([IEEE Access 2023](https://doi.org/10.1109/ACCESS.2023.3300726), abstract only). Any false-alarm figure here must therefore be scoped as "FAR on walk/sit/stand/hand/empty".

ESP32 fall evidence remains thin:

| Study | Result | Protocol |
|---|---|---|
| Ye et al., ESP32-S3 + LSTM | 89% on the fall class ([SPIE 2023](https://doi.org/10.1117/12.3010434), abstract only) | Not stated |
| TED-Net, 3-antenna ESP32-S3 | 90.5% | Within-subject |
| WiFall/KNN-MMD | 91.88% in-domain, 59.86% for a new person | Leave-one-person-out |
| Amritha et al. | 87.5% recall on 16 falls | Fall vs walk only |

Sources: [TED-Net](https://arxiv.org/html/2504.16655); [KNN-MMD](https://arxiv.org/html/2412.04783v5); [Amritha et al.](https://arxiv.org/html/2611.15532).

No surveyed ESP32 paper reports LOSO with both sensitivity and FAR. The proof of concept's 46.8% at 8.0% is therefore a modest but honest data point, and the ≥95% / ≤3% targets remain stretch goals.

### Imbalance, thresholds and the one-class route

The imbalance is mild: 79 falls against 100 trials per ADL class. Class weights plus a threshold set inside inner folds are therefore enough. On wearable falls, class weights kept 74.49% specificity, against 66.88% for SMOTE and 66.49% for focal loss. SMOTE plus threshold moving jumped to 89.98%, and that jump came from the threshold ([CDL-Fall](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512051/), random window split). Under subject-group cross-validation, SMOTE added only +0.57 F1 ([Tu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11991046/)).

A one-class novelty model trained on non-fall trials, as in SiFall and mmFall, is worth an arm of its own. It uses subject 3's data completely, because it needs no falls.

No Wi-Fi fall paper sweeps packet rate, so the minimum usable rate for falls has to be measured on SafeSense-5G's own data.

## A second ESP32 buys two to five points, and only with late or shared fusion

### Controlled comparisons find small gains

Where comparisons are controlled, a second receiver adds little:

| Study | Setting | Single receiver | Two receivers |
|---|---|---|---|
| OPERAnet ([Koupai et al.](https://ar5iv.labs.arxiv.org/html/2209.03765)) | Random split | 88.3% accuracy | 92.1% |
| Geometry Over Depth ([WiMob 2026](https://doi.org/10.1109/WiMob70978.2026.11777920), abstract only) | Two ESP32 nodes, 3 subjects | Best node 75.56% | Fused 80.0% |
| ESP32-S3 thesis ([GitHub](https://github.com/stef4k/thesis-wifi-human-activity-recognition), not peer-reviewed) | Unseen rooms | One well-placed link within 3-4 macro-F1 of two-link fusion | — |

Two further studies point the same way:

- **Exposing the CSI:** extra antennas did not help cross-subject accuracy, and cross-environment accuracy was near random ([PerCom 2023](https://ar5iv.labs.arxiv.org/html/2302.00992)).
- **Nine-node ESP32-C3 mesh:** concatenating all 72 links (AUC 0.489-0.501) did worse than the single best link (0.535-0.543) ([dilution study](https://arxiv.org/html/2602.10823v3)).

Only few-shot, cross-domain decision fusion reports large gains. FewSense reports +29.2% ([TMC](https://arxiv.org/abs/2203.02014), abstract only). ReWiS's "38% macro-diversity" figure is **unverified**: it comes from a search snippet ([arXiv 2201.00869](https://arxiv.org/abs/2201.00869)).

### Late fusion survives domain shift better

How fusion is done matters as much as whether it is done. In SHARP, decision-level fusion stayed robust while data-level fusion degraded out of domain ([SHARP](https://ar5iv.arxiv.org/html/2103.09924)):

| Test condition | Decision-level fusion | Data-level fusion |
|---|---|---|
| New day and person | 99.33% | 89.43% |
| Moved monitor and new person | 96.59% | 81.68% |
| New room | 95.99% | 88.02% |

More views can also hurt. On OPERAnet, fusing all 15 features scored below the best 7 ([Koupai et al.](https://ar5iv.labs.arxiv.org/html/2209.03765)).

### Receiver hardware and placement can mimic a fusion gain

Receiver heterogeneity is a measurable error source. Gain alignment recovers up to 75% of the accuracy lost to cross-receiver differences ([IoT-J 2026](https://doi.org/10.1109/JIOT.2026.3667804), abstract only).

Placement and antenna choice move single-link accuracy as much as a second link does:

- Repositioning an ESP32 receiver gave +29.4% ([Hernandez & Bulut 2020](https://doi.org/10.1109/WoWMoM49955.2020.00056)).
- A biquad antenna beat the built-in printed inverted-F antenna (PIFA) by 92.0% vs 86.8% through a wall ([ICIP 2024](https://ar5iv.labs.arxiv.org/html/2401.01388)).
- Antenna polarization alone moved fall accuracy from 50% to 92% ([Cardenas et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8347439/)).

A 1-vs-2 gap can therefore be a placement effect. RX1 and RX2 must each be reported as single-receiver baselines.

### Our single run sits in the expected range

Our proof of concept gave:

| Condition | Accuracy | AUC |
|---|---|---|
| RX1 only | 40.3% | 0.704 |
| RX1 + RX2 fused | 44.2% | 0.762 |

That is **+3.9 points of accuracy and +0.058 AUC**, inside the literature's range. It is still a single run: the RX2-only result is missing and there is no paired test.

There is also a structural question. Each receiver's dominant source is logged as its own MAC. If those streams are frames addressed to each receiver, RX1 and RX2 sample *different* frame sequences, possibly at different rates. Per-RX rate and missing-rate statistics belong in every 1-vs-2 table.

Exact per-frame pairing is not possible on the current firmware. It needs either ESP-IDF ≥5.5's `rx_seq` field, or the 802.11 header that ESP-IDF ≥5.2 exposes ([v5.5 header](https://raw.githubusercontent.com/espressif/esp-idf/v5.5/components/esp_wifi/include/local/esp_wifi_types_native.h)). Cross-RX phase is meaningless on stock boards.

### How to run the comparison

The comparison matrix from the earlier report still applies:

- RX1-only;
- RX2-only;
- early fusion;
- late fusion;
- OR/AND fall-decision fusion;
- a width control;
- an inner-CV-selected single receiver.

Add per-RX quiet-baseline normalization, applied equally to every condition.

The statistics must respect five folds. A Wilcoxon test over 5 subjects cannot reach p < 0.05. Instead, use McNemar on pooled trial-level predictions plus a cluster bootstrap on the differences.

## Subject 3's missing falls call for breaking a shortcut, not filling a cell

### The real risk is a shortcut, not a missing number

The missing cell is narrower than it looks. The folds that test subjects 1, 2, 4 and 5 already test falls the model has never seen. The real risk sits in folds that *train* on subject 3. There, everything specific to subject 3 (body size, position, static multipath) correlates perfectly with "not a fall". The model can then learn the shortcut "subject-3-like ⇒ non-fall". That shortcut is exactly how the system would fail for a future user who resembles subject 3. Copying another subject's falls into subject 3's slot is still both fabrication and leakage, as the earlier report showed.

### Augmentation helps little when the test person is unseen

Published augmentation gains shrink sharply once the test person is unseen:

| Study | Setting | Gain |
|---|---|---|
| RFBoost ([IMWUT 2024](https://arxiv.org/html/2410.07230v1)) | Cross-person | **+0.87%**, against +5.52% cross-receiver |
| SDMix ([IMWUT 2022](https://arxiv.org/html/2206.06629)) | Vanilla mixup, cross-person | +0.95 points (70.26% → 71.21%) |
| Tu et al. ([Sensors 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11991046/)) | Wrist falls, subject-group CV, full data | Transforms +0.27 F1; diffusion +2.13 F1, but precision fell from 91.64 to 85.79 |
| Leng et al. ([LOSO, 2025](https://arxiv.org/html/2506.07612v1)) | Classical augmentation | Mixed; one cell fell from 71.42 to 62.34 macro-F1 |

On ESP32-S3 data, many of Strohmayer & Kampel's augmentation gains fall within ±5-10 points of run-to-run spread, and they used no subject-wise split ([arXiv 2401.00964](https://arxiv.org/html/2401.00964v1)).

Two transform types carry specific risks:

- **Time warping** significantly hurt MLP and LSTM-FCN models across 128 datasets ([Iwana & Uchida](https://pmc.ncbi.nlm.nih.gov/articles/PMC8282049)).
- **Generators** can do outright harm. Unfiltered synthetic CSI degraded performance by 13.4% ([Gong et al., MobiSys 2025](https://arxiv.org/abs/2506.23174)). Synthetic falls hurt every method on the SmartFallMM wrist data ([Alamgeer et al.](https://arxiv.org/html/2505.04660v1)).

User-style generators exist, but none fits this project:

- CM-GR gains +9.29 points cross-subject on average, but needs the target user's video plus unlabelled CSI ([CM-GR](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473887/)).
- CsiGAN needs the left-out user's unlabelled data ([CsiGAN](https://www.github.com/ChunjingXiao/CsiGAN)).
- Neither is validated on ESP32 amplitude.

No paper studies a missing subject × class cell. Compositional held-out-combination splits are the closest template ([PMC9009598](https://pmc.ncbi.nlm.nih.gov/articles/PMC9009598/)).

### Our augmentation null matches the literature

Our measurements agree. Augmentation was applied to training folds only and provenance-tracked: 79 generic plus 79 "subject-3 energy-style" synthetic falls built from subjects 1, 2, 4 and 5. It **never moved fall-vs-ADL AUC outside 0.45-0.53**, and every confidence interval overlapped 0.5. It only moved the operating point: random-forest sensitivity rose from 3.8% to 26.6%, and false alarms rose from 9.2% to 31.5%.

That is a class-prior shift along a chance ROC. Threshold moving gives the same shift at no cost, as in the CDL-Fall result above. The test ran on the contaminated pipeline, where the features carried almost no fall signal. It must be repeated on v2 before augmentation is abandoned or adopted.

### Test augmentation with a leave-one-cell-out simulation

The only honest test of subject-style augmentation is a leave-one-cell-out simulation. For each subject j in {1, 2, 4, 5}:

1. Remove j's falls from training, but keep j's non-fall trials.
2. Train with and without j-style augmentation of the other subjects' falls.
3. Test on j's real falls plus held-out non-fall trials.

The same rules apply throughout:

- **Style ratios** are fitted in-fold, from training subjects only.
- **In subject 3's own test fold**, ratios computed from subject 3's non-fall data count as transductive adaptation. Report that result separately.
- **Copies** stay grouped with their source trial.
- **Class weights** are recomputed after augmentation, with copies down-weighted (weight 1/k for k copies of a fall).
- **Test folds** contain only real trials.

### If augmentation is used, and the alternatives

If augmentation is used at all, the evidence ranks the transforms as follows:

- **Use:** crop or shift that keeps the event inside the window. Avoid circular wrap, which splices post-fall lying onto pre-fall standing.
- **Use:** subcarrier dropout, applied after the 52-subcarrier mask.
- **Use with care:** noise at the measured empty-room floor, and mild ±25% scaling, which scale-free features largely cancel.
- **Use with care:** within-class mixup, and time warping capped at ±10%.
- **Avoid:** rotation, flips and contrast transforms.

Two routes sidestep the cell altogether:

- **A one-class ADL model** (SiFall-style) needs no falls.
- **A template built from other subjects** (DeFall-style) needs no falls from the user either.

Recording about 20 real mattress falls for subject 3 still beats every workaround.

## Sampling, furniture and tiers are the open problems this project can touch

### What the surveys name as unsolved

Across 15 surveys and position papers, the most-cited open problems are:

| Open problem | Surveys naming it |
|---|---|
| Robustness to environment or domain change | 11 |
| Data scarcity and benchmarks | 7 |
| Edge and real-time feasibility | 6 |
| Privacy | 5-6 |
| Sensing–communication coexistence and irregular sampling | 5 |
| Multi-user sensing | 4 |

Sources: [Tan et al.](https://arxiv.org/pdf/2111.07038); [Hernandez & Bulut](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf); [Zhu et al.](https://arxiv.org/html/2506.04322v1).

### Environment robustness is out of reach in one room

A one-room, five-subject project cannot address environment robustness. The cost of a new environment is large:

- Unseen environments cut accuracy from **82% to 42%**; gradient reversal recovered only to 63% ([Monjur et al.](https://arxiv.org/html/2402.09452v1)).
- Furniture or TX/RX moves cost ESP32 HAR up to 21% F1 ([WiFi-Wild](https://doi.org/10.1109/IEEEDATA.2026.3681245), abstract only).
- Rearranging furniture *within* a room cost presence detection only about 2 TPR points ([Sci. Rep. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10864388/)).

SafeSense-5G's bare/chair/mattress variants exist only in empty-room trials, recorded on a separate day with about 5 dB higher RSSI. The strongest furniture claim available is therefore narrow: empty-room detection is insensitive to three furniture states within one session. Even that claim holds only once a probe shows that RSSI or day alone does not separate the classes.

### Irregular sampling has moved beyond linear interpolation

The field now handles irregular sampling with three newer approaches:

- **Time-aware models.** FuseFi measured real 2.4 GHz traffic at a missing rate (MR) of 0.60 and a sampling coefficient of variation (SCV) of 1.68. Linear interpolation *lowered* its accuracy from 86.11% to 80.79%, and in synthetic sweeps an MR at or below 0.5 stayed at 84.9-86.8% ([FuseFi](https://arxiv.org/html/2512.22143)).
- **Learned imputation.** On loss-containing WiFall fall data, CSI-BERT2 scored 94.79%, against 52.99% for a CNN (75.78% with interpolation). However, recovery *lowered* the strongest model from 99.48% to 89-92% ([CSI-BERT2](https://arxiv.org/html/2412.06861v4)).
- **Rate-agnostic training.** A train/test rate mismatch dropped one model from 93.6% to 55.1%, which rate augmentation largely fixes ([SRV-NN](https://arxiv.org/html/2605.08308)).

On a controlled, constant 100 Hz ESP32 stream, interpolation barely mattered ([Hernandez & Bulut](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf)).

SafeSense-5G's ~70 Hz median implies an MR of at least 0.3 on a 100 Hz grid (our arithmetic), and bursts push it higher. No surveyed ESP32 paper publishes per-receiver gap distributions, so reporting them is a direct contribution.

### Tiered edge architectures lack measured accounting

Tier design is under-measured across the field:

- **Inference rate.** Only 11 of 176 surveyed papers report it ([Hernandez & Bulut](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf)).
- **Where the energy goes.** The same study's ESP32 currents are 58 mA idle and about +17 mA for inference, against 110-121 mA for active CSI reception and about 7 mA in deep sleep. **Tier 1's dormant skip therefore saves compute and uplink, not radio energy.**
- **Uplink savings.** Event-driven upload of compact features cut daily data per device from 1.41 GB to 10.36 MB in Origin's deployment, where raw 100 Hz CSI runs about 149 MB/h ([Zhu et al.](https://arxiv.org/html/2506.04322v1)).
- **Early exit.** No CSI early-exit network was found. A green-sensing survey only proposes cascaded heads ([Koo et al.](https://www.mdpi.com/1996-1073/19/2/573), snippet only).
- **5G offload.** No measured 5G-MEC offload of Wi-Fi sensing was found.

The 802.11bf standard was published on 26 September 2025. It makes uncompressed CSI its only sub-7 GHz feedback type ([IEEE SA](https://standards.ieee.org/ieee/1641/10365/); [Du et al.](https://arxiv.org/pdf/2310.17661)). That makes uplink accounting timely, while the project's 802.11n ESP32 setup is pre-11bf.

## Citable paper tables, deduplicated and graded by verification

**How the tables are organized.** Each paper appears once, in the table where it is most relevant. Later tables cross-reference it by row ID, such as "see T1.3".

**Verification codes.**

- **FT**: numbers were read in the full text or HTML. Some rows were read through a fetch-and-summarise tool, so spot-check them before publication.
- **ABS**: from the abstract or metadata only.
- **SEC**: numbers come from another paper's table or reimplementation.
- **SNIP**: from a search snippet or secondary summary only. These numbers are **unverified; do not cite them without checking the source**.
- **NPR**: not peer-reviewed (repository, thesis or blog).
- **DOI?**: the DOI, volume or venue is not confirmed.

**Earlier-round rows.** Rows from the earlier research round carry the verification level recorded at that time.

### T1. Cross-subject generalization, few-shot calibration and physics features (non-adversarial)

| ID | Paper | Venue, year | DOI / link | What it shows (protocol) | Verif. |
|---|---|---|---|---|---|
| T1.1 | SHARP — Meneghello, Garlisi, Dal Fabbro, Tinnirello, Rossi | IEEE TMC 22(10):6160-6175, 2023 | [10.1109/TMC.2022.3185681](https://ar5iv.arxiv.org/html/2103.09924) | Nexmon, 80 MHz, 4 RX antennas, 3 people. Trained on 1 person: 99.33% new person+day; 95.99% new room+person. Late fusion 96.59% vs early 81.68% | FT |
| T1.2 | CrossFi — Zhao, Chen, Cai et al. | arXiv 2408.10919 v4, 2025; IEEE IoT-J early access (venue SNIP) | [arXiv](https://arxiv.org/html/2408.10919v4) | ESP32-S3 1×1, 100 Hz, 8 people (gestures). Zero-shot 64.81% (ResNet-18 40.84%); 1-shot 84.47% (91.72% with MK-MMD); near in-domain at about 5 shots | FT |
| T1.3 | KNN-MMD + WiFall dataset — Zhao, Cai, Chen et al. | arXiv 2412.04783 v5, 2025; part at ICCC 2025; arXiv page lists IEEE TMC DOI 10.1109/TMC.2025.3644902 (DOI?) | [arXiv](https://arxiv.org/html/2412.04783v5) | ESP32-S3 + router, 10 people, binary fall detection. In-domain 91.88%; zero-shot 59.86%; DANN 74.06%; MK-MMD 72.26%; ADDA 62.81%. 1-shot: KNN-MMD 77.62%, CrossFi 80.93%, fine-tune 84.03%, plain kNN 49.63%. Few-shot baselines report best test accuracy | FT |
| T1.4 | CrossGR — Li, Chang, Song, Wang, Chen, Tang, Wang | Proc. ACM IMWUT 5(1), 2021 (DOI?) | [PDF](https://eprints.whiterose.ac.uk/170933/6/ubicomp2021.pdf) | Intel 5300. CNN 95%→36% on unseen users; SVM/KNN/RF on handcrafted statistics also poor; t-SNE clusters by user. 60 users: 87.4%. Synthetic data 73%→91%; about +14 points from 36→48 training users | FT |
| T1.5 | DeFall — Hu, Zhang, Wu, Wang, Liu | IEEE IoT-J 2021/22; ICASSP 2020 version | [10.1109/JIOT.2021.3116136](https://originwirelessai.com/wp-content/uploads/2024/04/DeFall.pdf) | Intel 5300, 3 antennas, 5 GHz. Speed/acceleration features with DTW against a dummy-built template. 91.67-100% detection per user (10 users); >95% detection at <1.5% FAR overall. New environments 95.37/96.67% vs WiFall 56.11/52.78% | FT |
| T1.6 | Ma, Arshad, Muniraju, Torkildson, Rantala, Doppler, Zhou | ACM Trans. IoT, 2021 (DOI?) | [PDF](https://gzhou.pages.wm.edu/wp-content/blogs.dir/5736/files/sites/13/2017/12/ACM_TIOT_2020.pdf) | Intel 5300, 7 people. Leave-one-person-out: CNN / +RL / +state machine 62 / 77 / 97%. Separable CNN 84% on unseen persons. FallDeFi data 51 / 64 / 83% | FT |
| T1.7 | WiGr — Zhang, Tang, Yin, Ni | IEEE IoT-J 9(11):8584-8596, 2022 | [10.1109/JIOT.2021.3114309](https://zhang-xie.github.io/papers/WiGr_zhang.pdf) | Prototypical network. Cross-user: >90% on CSIDA (5 users), ≤89% at 4-shot on Widar3.0. Cross-environment 60 / 65 / 75 / 83% at 1-4 shots | FT |
| T1.8 | Wi-Learner — Feng et al. | Proc. ACM IMWUT 6(3), 2022 | [10.1145/3550318](https://api.crossref.org/works/10.1145%2F3550318) | 1-shot cross-domain 74.2-94.9% (Widar3.0) | ABS |
| T1.9 | FewSense — Yin, Zhang, Shen, Chen | IEEE TMC 23(1):453-468, 2024 (online 2022) | [10.1109/TMC.2022.3221902](https://arxiv.org/abs/2203.02014) | 5-shot novel classes in unseen domains: 82.7-96.5%. Multi-receiver fusion +29.2%. SignFi figure given as 90.3% in one source and 93.9% in another (conflict) | ABS |
| T1.10 | OneFi — Xiao, Liu, Han, Ren | ACM SenSys 2021, pp. 206-219 | [10.1145/3485730.3485936](https://ruixiao24.github.io/files/rui_onefi.pdf) | Unseen gestures 84.2 / 94.2 / 95.8 / 98.8% at 1 / 3 / 5 / 7 shots; virtual samples. No cross-user number | ABS/SNIP |
| T1.11 | RF-Net — Ding, Chen, Zheng, Luo | ACM SenSys 2020, pp. 517-530 | [arXiv 2111.04566](https://arxiv.org/pdf/2111.04566) | Metric meta-learning; no per-person split. Third-party Widar cross-location result: 55.23 / 64.02% at 1 / 3 shots | FT; SEC |
| T1.12 | MatNet-eCSI — Shi, Zhang, Xu, Cheng | IEEE TMC 21(2):540-554, 2022 | [10.1109/TMC.2020.3012433](https://opus.lib.uts.edu.au/handle/10453/146210) | One-shot adaptation to a new environment; numbers not retrieved | ABS |
| T1.13 | Jiang, Yan, Wang, Chou, Hu | arXiv 2511.18792, 2025 | [arXiv](https://arxiv.org/html/2511.18792v1) | 14 corpora, none ESP32. Widar cross-subject: supervised 54.7%, MAE fine-tune 67.5%, linear probe 46.1%. Pre-training on about 1.3k samples −5.5 points; parity at about 13.2k | FT |
| T1.14 | CSI-Bench — Zhu et al. | NeurIPS 2025 Datasets & Benchmarks | [10.52202/085713-5648](https://arxiv.org/html/2505.21866v2) | 35 users, 16 device types incl. ESP32-S3. HAR 88.06%→54.72% cross-user; cross-device 61.8%. Fall subset (17 users) evaluated in-distribution only, 92.2-94.9% | FT |
| T1.15 | TCN-MAML — Lin et al. | Sensors 25(13):4216, 2025 | [10.3390/s25134216](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252493/) | Supervised 99.6%→67.4% on unseen pairs. The MAML 99.6% figure leaks target subjects | FT |
| T1.16 | Wi-AM — Xie, Li, Feng, Lin, Meng | Sensors 24(5):1354, 2024 | [10.3390/s24051354](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934180/) | Hybrid: adversarial extractor plus prototypes. Cross-person 78.42 / 83.6% at 1 / 3 shots | FT |
| T1.17 | Liu, Pan, Zeng, Ren, Wang, Kong, Wang | arXiv 2512.04521, 2025 | [arXiv](https://arxiv.org/pdf/2512.04521) | Cross-antenna DFS with an attention CNN; leave-one-user-out 98.92% on 14 Widar users. Needs ≥2 antennas on one radio | FT |
| T1.18 | AutoFi — Yang, Chen, Zou, Wang, Xie | IEEE IoT-J 10(8):7416-7425, 2023 | [10.1109/JIOT.2022.3228820](https://arxiv.org/abs/2205.01629) | On ESP32 new-person 1-shot (measured by T1.3): fall 48.05%, gesture 27.05% | SEC |
| T1.19 | Xu, Wang, Zhu, Zheng (SSL study) | arXiv 2308.02412, 2023 (venue?) | [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02412) | Linear evaluation, supervised vs MAE vs SimCLR: 98.03 / 84.29 / 83.93 (UT-HAR); 86.33 / 69.24 / 40.43 (Widar) | FT (partial) |
| T1.20 | WiHF — Li et al. | IEEE INFOCOM 2020, pp. 586-595; IEEE TMC 21(2):757-768 | [listing](https://www.bohrium.com/paper-details/wihf-gesture-and-user-recognition-with-wifi/812587223990927361-2468) | Handcrafted motion-change pattern ≈ Widar3.0 cross-domain accuracy at about 30× less compute; no per-user number | ABS/SNIP |
| T1.21 | Alsaify, Almazari, Alazrai, Alouneh, Daoud | Applied Sciences 12(2):930, 2022 | [10.3390/app12020930](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/app12020930?fields=title,authors,abstract,year,venue,externalIds) | Handcrafted features + SVM, 91.27% average; LOSO number not retrieved | ABS |
| T1.22 | MetaSense — Gong, Kim, Shin, Lee | ACM SenSys 2019, pp. 110-123 | [PDF](https://nmsl.kaist.ac.kr/pdf/SenSys19_MetaSense.pdf) | Meta-learning for a new user from 1-2 examples per class; not Wi-Fi | ABS |
| T1.23 | Ye, Xu, He, Yin, Ohtsuki, Gui | Bioengineering 11(11):1124, 2024 | [10.3390/bioengineering11111124](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591964) | 2 ESP32s (STA ping / AP reply), 100 pkt/s, 5 subjects. Same-person 91-97%; LOSO ≈77% (CNN), ≈83.9% (best), means computed from the paper's table. Fall accuracy 3.82% in one fold | FT |
| T1.24 | Varga | Sensors 24(24):8201, 2024 | [10.3390/s24248201](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679234/) | 2D-CNN, random vs subject split: 92.2% vs 66.4% | FT |
| T1.25 | Varga (ImgFi leakage) | Sensors 24(10):3159, 2024 | [10.3390/s24103159](https://pmc.ncbi.nlm.nih.gov/articles/PMC11124867/) | Precision, random vs subject split: 99.0% vs 23.4% (WiAR); 99.5% vs 47.4% (Widar) | FT |
| T1.26 | FuseFi / UniFi — Dong, Yang, Srivastava | IEEE IoT-J 2026 (accepted), DOI 10.1109/JIOT.2026.3731514 | [arXiv 2512.22143](https://arxiv.org/html/2512.22143) | LOSO 96.9%→41.6%; linear head on 5 / 10 windows per class 91.3 / 94.7%. 2.4 GHz traffic MR 0.60, SCV 1.68; linear interpolation 86.11%→80.79% | FT |
| T1.27 | Bayik, Ajayi, Roggen, Birch | arXiv 2604.16572, 2026 | [arXiv](https://arxiv.org/html/2604.16572v1) | WiMANS identity-dependent macro-F1 80.38→32.61 on unseen users | FT |
| T1.28 | Custance, Khan, Parkinson | Electronics 12(21):4500, 2023 | [10.3390/electronics12214500](https://doi.org/10.3390/electronics12214500) | ESP32 sit/stand CSI identifies 15 people at 84.29% / 74.13% | ABS |
| T1.29 | Elkelany, Ross, McKeever | AICS 2024, CEUR-WS Vol-3910 | [PDF](https://ceur-ws.org/Vol-3910/aics2024_p78.pdf) | Argues LOSO should be standard; numbers not extracted | ABS |
| T1.30 | Widar3.0 — Zheng et al. | ACM MobiSys 2019; IEEE TPAMI 44(11), 2022 | [10.1145/3307334.3326081](https://cswu.me/papers/mobisys19_widar3_paper.pdf) | BVP. 92.7% in-domain; 82.6-92.4% cross-domain; sharp accuracy drop below 3 links | SNIP/SEC |
| T1.31 | Shahbaz et al. (ESP32 TinyML) | IEEE Access 2026 per repo, DOI 10.1109/ACCESS.2026.3742666 (DOI?) | [GitHub](https://github.com/ahmadshahbazzz/esp32-csi-har-frontier) | Leave-one-user-out mean −29 points; CSI-Bench 84.1%→51.7%; RF/DT run in 1-2 ms on a classic ESP32 | NPR/ABS |
| T1.32 | gladiator-13 ESP32 HAR benchmark | GitHub | [repo](https://github.com/gladiator-13/ESP32-WiFi-CSI-Human-Activity-Recognition) | 37 subjects, 20 classes; RF best in every subject-disjoint setting | NPR |
| T1.33 | SenseFi — Yang et al. | Patterns 4(3):100703, 2023 | [10.1016/j.patter.2023.100703](https://arxiv.org/html/2207.07859v3) | 92-99.7% on random 8:2 splits; no ESP32 data; deep models overfit cross-domain | FT |
| T1.34 | Yousefi et al. | IEEE Commun. Mag., 2017 | [10.1109/MCOM.2017.1700082](https://ar5iv.labs.arxiv.org/html/1708.07129) | Per-class recall LSTM ≈0.91 vs RF ≈0.58 (approximate; protocol unstated). Frame-rate selection named as an open problem | FT (approx.) |
| T1.35 | LiteHAR — Salehinejad & Valaee | IEEE ICASSP 2022 | [arXiv 2201.09310](https://arxiv.org/abs/2201.09310) | Random kernels + Ridge; no numbers in abstract | ABS |
| T1.36 | CARM — Wang et al. | ACM MobiCom 2015 | [PDF](https://old.sigmobile.org/mobicom/2015/papers/p65-wangA.pdf) | CSI-speed model; "<300 Hz" guidance | SNIP |

### T2. Adversarial and discrepancy-based domain adaptation / generalization

| ID | Paper | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T2.1 | DANN — Ganin et al. | JMLR 17(59), 2016 | [PDF](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf) | Shallow network on fixed features: +0.2 / +1.0 point mean over 12 pairs. MNIST→SVHN failure; λ schedule γ = 10 | FT |
| T2.2 | Zhao, Tachet des Combes, Zhang, Gordon | ICML 2019, PMLR 97:7523-7532 | [PMLR](https://proceedings.mlr.press/v97/zhao19a.html) | Lower bound: invariant representations cannot fix label shift | ABS |
| T2.3 | DomainBed — Gulrajani & Lopez-Paz | ICLR 2021 | [arXiv 2007.01434](https://ar5iv.labs.arxiv.org/html/2007.01434) | ERM 67.0, CORAL 67.2, MMD 66.8, DANN 66.4. About 7-point oracle-selection gap (SNIP) | FT; SNIP |
| T2.4 | EI — Jiang et al. | ACM MobiCom 2018 | [10.1145/3241539.3241548](https://cse.buffalo.edu/~lusu/papers/MobiCom2018.pdf) | 22 subject-room source domains: 0.75; without balance constraint 0.61; "low" with 2 domains; RF baseline did not improve | FT |
| T2.5 | WiADG — Zou et al. | IEEE ICCCN 2018 | [10.1109/ICCCN.2018.8487345](https://api.openalex.org/works?filter=title.search:unsupervised%20adversarial%20domain%20adaptation%20gesture) | +25% average in new environments | ABS |
| T2.6 | JADA — Zou et al. | IEEE ICMLA 2018 | 10.1109/ICMLA.2018.00037 | 98.75% in the source environment; adapted accuracy not reported | ABS |
| T2.7 | WiCAR — Wang, Liu, Gong | IWQoS 2019 | [10.1145/3326285.3329054](https://unpaywall.org/10.1145%2F3326285.3329054) | About 53% (no DA) → 83% (prior DA) → 95% | ABS (snippet) |
| T2.8 | MobileDA — Yang et al. | IEEE IoT-J 7(8):6909-6918, 2020 | 10.1109/JIOT.2020.2976762 | Server→edge cross-domain distillation; no numbers | ABS |
| T2.9 | CsiGAN — Xiao, Han, Ma, Qin | IEEE IoT-J 6(6):10191-10204, 2019 | [10.1109/JIOT.2019.2936580](https://www.github.com/ChunjingXiao/CsiGAN) | Leave-one-user-out with 10% unlabelled left-out data (SignFi, FallDeFi); numbers not retrieved | ABS |
| T2.10 | RF-Sleep — Zhao, Yue, Katabi, Jaakkola, Bianchi | ICML 2017, PMLR 70:4100-4109 | [PDF](http://proceedings.mlr.press/v70/zhao17d/zhao17d.pdf) | Conditional adversary, 25 subjects: 79.8%, κ 0.70, vs a different prior system at 64% | FT |
| T2.11 | FiDo — Chen et al. | WWW 2020 | 10.1145/3366423.3380091 | +11.8% average F1 for new users | ABS |
| T2.12 | DAFI — Li et al. | Proc. ACM IMWUT 5(4), 2021 (DOI?) | [researchr](https://researchr.org/publication/LiCWWL21/related) | Up to +14.2% | ABS |
| T2.13 | Narui, Shu, Gonzalez-Navarro, Ermon | W3PHAI 2019 / Springer 2020 | [10.1007/978-3-030-24409-5_17](https://doi.org/10.1007/978-3-030-24409-5_17) | Unsupervised DA for Wi-Fi fall detection in new rooms; no numbers | ABS |
| T2.14 | Nguyen & Nguyen | CISS 2020 | [10.1109/CISS48834.2020.1570617398](https://ar5iv.labs.arxiv.org/html/2005.11932) | FallDeFi data, leave-one-domain-out: CNN-ADA 41.5-64.1%, LSTM-ADA 40.2-66.0%, VADA 52.7%; gain "not significant" | FT |
| T2.15 | Zhang, Zhang, Li, Hu, Chen (USTC) | arXiv 2111.10602, 2021 | [PDF](https://arxiv.org/pdf/2111.10602) | Widar cross-subject: CrossSense 81.77, EI 83.61, proposed 88.96 | FT |
| T2.16 | Hassan, Kelsey, Rahman | PLOS ONE 2024 | [10.1371/journal.pone.0298888](https://research-repository.st-andrews.ac.uk/bitstream/10023/29720/1/Hassan_2024_PONE_Adversarial-AI_CC.pdf) | 3 users: GRL+MMD 69.42 vs MMD 59.83 micro-F1; spread 40-85 | FT |
| T2.17 | DATTA — Strohmayer, Sterzinger, Wödlinger, Kampel | arXiv 2411.13284, 2024 | [HTML](https://arxiv.org/html/2411.13284) | F1: source-only 40.62, DAT 42.02, augmentation 49.32, DAT + augmentation 65.66, DATTA about 68 | FT |
| T2.18 | GesFi — Zhang et al. | Proc. ACM IMWUT 2026 | [arXiv 2601.03825](https://arxiv.org/html/2601.03825v2) | 1-link cross-user: 93.67 vs SelfReg 53.10 and WiSGP 56.77; "manifold distortion" critique | FT (partial) |
| T2.19 | AirFi — Wang, Yang, Cui, Xie, Sun | arXiv 2209.10285, 2022 | [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.10285) | Unseen room: CNN 29-37% vs 88.8-92.4%; with 10 labelled target samples 91.7-96.1% | FT |
| T2.20 | Wi-CBR — Zhang et al. | arXiv 2506.11616, 2025 | [HTML](https://arxiv.org/html/2506.11616) | Widar mean 97.17 vs WiSR 63.41 | FT; SEC |
| T2.21 | WiSGP | venue? | [A*STAR](https://oar.a-star.edu.sg/communities-collections/articles/19399) | 1-link cross-user 56.77 (as reported in T2.18) | SEC |
| T2.22 | CrossSense — Zhang et al. | ACM MobiCom 2018 | [via CSUR](https://dl.acm.org/doi/10.1145/3570325) | 100 participants; reimplemented cross-subject 81.77 (in T2.15) | SEC |

### T3. Fall detection systems (Wi-Fi/RF), with radar contrast

| ID | Paper | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T3.1 | SiFall — Ji, Xie, Li | ACM SenSys 2022 | [10.1145/3560905.3568517](https://arxiv.org/html/2301.03773v2) | 2 Atheros APs, 2.4 GHz, 20 MHz, 200 pkt/s; 16 subjects. Anomaly model trained on normal activities only: 100% TPR, 1.8% FPR over 19.3 h. RT-Fall on the same data: 64.9% TPR, 49.2% FPR. New environment: 78.3%. Alarm in 1.67 s | FT |
| T3.2 | FallDar — Yang, Zhang, Zhang | IEEE TMC 2022 | [10.1109/TMC.2022.3188779](https://doi.org/10.1109/TMC.2022.3188779) | Body speed + generative fall simulation + user-independent features. 5.7% FAR, 3.4% miss over 6 months | ABS |
| T3.3 | TCS-Fall — Zhou et al. | Digital Health (SAGE) 2024 | [10.1177/20552076241259047](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11151769/fullTextXML) | Intel 5300, about 1 kHz, 20 volunteers, 8 s windows. Train on 2, test on 18: AUC 0.999. No lie-down or pick-up negatives; one room | FT |
| T3.4 | FallAware — Regani, Wang, Hu, Zhu, Liu | IEEE JSAS 2025; ICASSP 2025 | [10.1109/JSAS.2024.3520517](https://doi.org/10.1109/JSAS.2024.3520517); 10.1109/ICASSP49660.2025.10890231 | 94.1% detection in unseen environments; <5 false alarms per month over 21 months | ABS |
| T3.5 | AmFall — Kim, Jeon, Jeong | IEEE IoT-J 2025 | [10.1109/JIOT.2025.3585401](https://doi.org/10.1109/JIOT.2025.3585401) | Amplitude-only CWT scalograms; 95.28-99.67% through-the-wall and cross-domain | ABS |
| T3.6 | Nakamura, Bouazizi, Yamamoto, Ohtsuki | IEEE IoT-J 2022; Globecom 2020 | [10.1109/JIOT.2022.3152315](https://doi.org/10.1109/JIOT.2022.3152315) | Fixed sliding windows (no segmentation); >0.92 accuracy across 2 rooms | ABS |
| T3.7 | TL-Fall — Zhang, Wang, Liu | IEEE SECON 2019 | [10.1109/SAHCN.2019.8824989](https://doi.org/10.1109/SAHCN.2019.8824989) | Few-shot transfer to a new environment: 86.83% sensitivity, 84.71% specificity | ABS |
| T3.8 | AFall — Chen et al. | IEEE TMC 2022 | [10.1109/TMC.2022.3157666](https://doi.org/10.1109/TMC.2022.3157666) | Two orthogonal receivers, MUSIC angle of arrival, training-free; 84.31% across 5 environments | ABS |
| T3.9 | Bayad et al. | IEEE MeMeA 2025 | [10.1109/MeMeA65319.2025.11068083](https://doi.org/10.1109/MeMeA65319.2025.11068083) | 87% in an unseen environment; 79% for new individuals; augmentation +13% | ABS |
| T3.10 | Preechakarnjanadit & Taparugssanagorn | IntechOpen ACRT 2025 | [10.5772/acrt.20250099](https://doi.org/10.5772/acrt.20250099) | CNN-LSTM 94.85%; lying-down negatives; 2.3 s latency. Subject counts are inconsistent | ABS |
| T3.11 | TED-Net + DGNN — Cho et al. | IEEE Sensors J. 2025 | [10.1109/JSEN.2025.3593126](https://arxiv.org/html/2504.16655) | Custom ESP32-S3 with 3 RX antennas, about 30 Hz effective, 20 subjects. Fall class 90.5%; fall→squat 6.8%; within-subject only | FT |
| T3.12 | Ye G. et al. | Proc. SPIE (CNIoT 2023) | [10.1117/12.3010434](https://doi.org/10.1117/12.3010434) | ESP32-S3, 52 subcarriers, LSTM. 96.03% overall, 89% on the fall class; split not stated | ABS |
| T3.13 | Amritha S. S. et al. | arXiv 2611.15532, 2026 | [arXiv](https://arxiv.org/html/2611.15532) | ESP32 pair, 5 s windows, 10 Hz low-pass. 93.1% accuracy, 87.5% recall (16 falls, fall vs walk) | FT |
| T3.14 | Wang, Pan, Liu, Li, Ren, Wang, Wang | IEEE TMC 2026, DOI 10.1109/TMC.2026.3742806 | [arXiv 2605.00869](https://arxiv.org/html/2605.00869) | Intel 5300 at 1 kHz, 22 volunteers. Unseen rooms 88.3 / 91.7% (vs 69.0 / 72.5%); live sit/stand 86.7%; 12.4 ms per window. No cross-subject test or DANN baseline | FT |
| T3.15 | WiFall — Han, Wu, Wang, Ni; Wang, Wu, Ni | IEEE INFOCOM 2014; IEEE TMC 2017 | [10.1109/INFOCOM.2014.6847948](https://doi.org/10.1109/INFOCOM.2014.6847948); [10.1109/TMC.2016.2557792](https://doi.org/10.1109/TMC.2016.2557792) | Same domain: 87% precision / 18% FAR; 90-94% / 13-15% FAR (TMC) | SEC |
| T3.16 | Anti-Fall — Zhang, Wang, Wang, Ma | ICOST 2015 | [10.1007/978-3-319-19312-0_15](https://doi.org/10.1007/978-3-319-19312-0_15) | About 89% / 13% FAR | SNIP |
| T3.17 | RT-Fall — Wang, Zhang et al. | IEEE TMC 16(2):511-526, 2017 | [10.1109/TMC.2016.2557795](https://doi.org/10.1109/TMC.2016.2557795) | Antenna phase difference; "sharp power decline" segmentation; +14 points sensitivity vs WiFall | ABS |
| T3.18 | FallDeFi — Palipana, Rojas, Agrawal, Pesch | Proc. ACM IMWUT 1(4), 2018 | [10.1145/3161183](https://doi.org/10.1145/3161183) | STFT features. 93% in the trained environment, about 80% in changed ones. Reimplemented in T3.14: 80.8% pooled, 68.2% NLoS | ABS; SEC |
| T3.19 | WmFall — Yang, Xiong, Shao, Niu | IJDSN 2018 | [10.1177/1550147718805718](https://doi.org/10.1177/1550147718805718) | 89% precision / 8% FAR | ABS |
| T3.20 | XFall — Chi et al. | IEEE JSAC 2024 | [10.1109/JSAC.2024.3413997](https://doi.org/10.1109/JSAC.2024.3413997) | Speed profile. 96.8% accuracy, 3.3% FAR over a year. Best F1 at a 1.5 s window (SNIP) | ABS |
| T3.21 | Chu, Cumanan, Sankarpandi, Smith, Dobre | IEEE Access 11:83763-83780, 2023 | [10.1109/ACCESS.2023.3300726](https://doi.org/10.1109/ACCESS.2023.3300726) | >700 samples, 4 rooms; >96% pooled. More comprehensive datasets give lower accuracy | ABS |
| T3.22 | DP-AcE | arXiv 2401.00160, 2024 | [arXiv](https://arxiv.org/pdf/2401.00160) | Shortening the window from 120 to 40 samples: TPR 87.49%→73.00%, FPR 8.92%→18.42% | Earlier round (level not recorded) |
| T3.23 | Cárdenas, Gutierrez, Aguilar-Ponce | Sensors 21(15):5121, 2021 | [10.3390/s21155121](https://pmc.ncbi.nlm.nih.gov/articles/PMC8347439/) | SDR with a Wi-Fi-like probe. HH polarization 92% vs VV 50% | ABS |
| T3.24 | Taha et al. | IEEE SENSORS 2021 | [10.1109/SENSORS47087.2021.9639715](https://doi.org/10.1109/SENSORS47087.2021.9639715) | SDR CSI + RF. Same classes as SafeSense minus gesture; 95.6% (10-fold) | ABS |
| T3.25 | Damodaran, Haruni, Kokhkharova, Schäfer | CCF TPCI 2020 | [10.1007/s42486-020-00027-1](https://doi.org/10.1007/s42486-020-00027-1) | Negatives: walk, sit, stand, run; "high accuracy" | ABS |
| T3.26 | Guo, Li, Han, Liu | IJERPH 20(6):4998, 2023 | [10.3390/ijerph20064998](https://doi.org/10.3390/ijerph20064998) | Routers on construction sites, 6 workers, 99%. The "degradation near 50 Hz" claim is unverified | ABS; SNIP |
| T3.27 | Moshiri, Shahbazian, Nabati, Ghorashi | Sensors 21(21):7225, 2021 | [10.3390/s21217225](https://doi.org/10.3390/s21217225) | Raspberry Pi / Nexmon, 7 activities incl. fall, about 95%, random split. Activity list unverified | ABS |
| T3.28 | Wang L. et al. | IEEE Sensors J. 2025 | [10.1109/JSEN.2025.3547060](https://doi.org/10.1109/JSEN.2025.3547060) | IMU trigger + CSI post-fall mobility check, 99% | ABS |
| T3.29 | Abdelfattah et al. | ITC-Egypt 2025 | [10.1109/ITC-Egypt66095.2025.11186708](https://doi.org/10.1109/ITC-Egypt66095.2025.11186708) | Wearable trigger → Wi-Fi SVM; claimed FAR 0.05%, datasets unspecified | ABS |
| T3.30 | BeamFall; BFI-Fall — Feng et al. | IEEE IoT-J 2026; IEEE SECON 2026 | [10.1109/JIOT.2026.3750395](https://doi.org/10.1109/JIOT.2026.3750395); [10.1109/SECON68281.2026.11578995](https://doi.org/10.1109/SECON68281.2026.11578995) | Beamforming feedback, 3 subjects; fall recall 98.1% / 95.8% | ABS |
| T3.31 | Cao, Liu, Li | IEEE IAEAC 2017 | [10.1109/IAEAC.2017.8054245](https://doi.org/10.1109/IAEAC.2017.8054245) | Dynamic time window; 91% recognition, FAR "0.06%" (implausible) | ABS |
| T3.32 | KS-FALL — Hao, Duan, Dang, Xu | IOP Conf. Ser. MSE 2019 | [10.1088/1757-899X/569/3/032068](https://doi.org/10.1088/1757-899X/569/3/032068) | Atheros, 5 GHz; 92.3%; squat negatives | ABS |
| T3.33 | Wang C. et al. | Tsinghua Sci. Technol. 2022 | [10.26599/TST.2022.9010011](https://doi.org/10.26599/TST.2022.9010011) | LOF segment + SVM / XGBoost, >90%. "Fall lasts about 1 s, 3 s window" (SNIP) | ABS |
| T3.34 | Cheng X., Huang, Zong | IEEE ICECE 2021 | [10.1109/ICECE54449.2021.9674346](https://doi.org/10.1109/ICECE54449.2021.9674346) | GMM-HMM on phase difference; real time | ABS |
| T3.35 | FallSense — Huang et al. | IEEE ICC 2019 | [10.1109/ICC.2019.8762032](https://doi.org/10.1109/ICC.2019.8762032) | Dynamic template matching, updated during use | ABS |
| T3.36 | Leelatien & Pararach | J. KMUTNB 2022 | [10.14416/j.kmutnb.2022.10.009](https://doi.org/10.14416/j.kmutnb.2022.10.009) | Slip severity, SVM 97.6-98.3% | ABS |
| T3.37 | Chagas, Lovisolo, Tcheou, Souza Filho | Eng. Appl. AI 2024 | [10.1016/j.engappai.2024.108628](https://doi.org/10.1016/j.engappai.2024.108628) | Bed-exit fall-risk anomaly detection | ABS |
| T3.38 | Hu, Li, Cheng, Zhang, Han, Umer | Dev. Built Environ. 2025 | [10.1016/j.dibe.2025.100745](https://doi.org/10.1016/j.dibe.2025.100745) | Smartphone receiver, Doppler, >93% | ABS |
| T3.39 | Stenbom | Halmstad Univ. thesis 2026 | [record](https://api.openalex.org/works?search=Fall%20Detection%20using%20Random%20Forest%20under%20Wi-Fi%20Interference) | ESP32-C3 + RF; sensing traffic cut download speed 23.43% and raised latency 15.79% | ABS/NPR |
| T3.40 | Sahoo (A. K.), Akhil, Udgata | LNNS (Springer) 2022 | [10.1007/978-981-19-0901-6_41](https://doi.org/10.1007/978-981-19-0901-6_41) | ESP32 through-wall presence + fall; numbers not retrieved | Title only |
| T3.41 | Fall papers with no extractable numbers | various | Ding & Wang TCE 2020 [10.1109/TCE.2020.3021398](https://doi.org/10.1109/TCE.2020.3021398); Li B. HPCC 2022 [10.1109/HPCC-DSS-SmartCity-DependSys57074.2022.00277](https://doi.org/10.1109/HPCC-DSS-SmartCity-DependSys57074.2022.00277); Cheng H. ICCE-TW 2019 [10.1109/ICCE-TW46550.2019.8991919](https://doi.org/10.1109/ICCE-TW46550.2019.8991919); Lai ICS 2024 [10.1109/ICS64339.2024.00028](https://doi.org/10.1109/ICS64339.2024.00028); FallDeWideo ISACom 2023 [10.1145/3615984.3616501](https://doi.org/10.1145/3615984.3616501); Keaton ICE/ITMC 2020 [10.1109/ICE/ITMC49519.2020.9198407](https://doi.org/10.1109/ICE/ITMC49519.2020.9198407); Mattela 2022 [10.1007/s42979-022-01111-2](https://doi.org/10.1007/s42979-022-01111-2); Xia & Chong 2023 [10.1016/j.compeleceng.2023.108763](https://doi.org/10.1016/j.compeleceng.2023.108763); He 2024 [10.1007/s11276-023-03625-w](https://doi.org/10.1007/s11276-023-03625-w); Zhou 2026 [10.1007/978-3-032-10554-7_3](https://doi.org/10.1007/978-3-032-10554-7_3); Xu 2026 [10.23919/ACCTCSCPS00012.2026.00010](https://doi.org/10.23919/ACCTCSCPS00012.2026.00010); Yang 2026 [10.2139/ssrn.7447197](https://doi.org/10.2139/ssrn.7447197); Do 2026 [10.1007/978-3-032-00972-2_10](https://doi.org/10.1007/978-3-032-00972-2_10); Nasser 2024 [10.1109/ICSPIS63676.2024.10812620](https://doi.org/10.1109/ICSPIS63676.2024.10812620); Chen Y. 2024 [10.1109/AIoTSys63104.2024.10780526](https://doi.org/10.1109/AIoTSys63104.2024.10780526); Chen Yunwang 2024 [10.1109/ROBIO64047.2024.10907570](https://doi.org/10.1109/ROBIO64047.2024.10907570) | Listed for completeness | ABS / title only |
| T3.42 | Aryokee — Tian, Lee, He, Hsu, Katabi | Proc. ACM IMWUT 2018 | [10.1145/3264947](https://doi.org/10.1145/3264947) | FMCW radar; >140 people, 57 environments; 94% recall, 92% precision; CNN + state machine | ABS |
| T3.43 | mmFall — Jin, Sengupta, Cao | IEEE TASE 2020 | [10.1109/TASE.2020.3042158](https://doi.org/10.1109/TASE.2020.3042158) | mmWave; ADL-only anomaly model + centroid drop; 98% of 50 falls, 2 false alarms | ABS |
| T3.44 | LT-Fall — Zhang D. et al. | Proc. ACM IMWUT 2023 | [10.1145/3580835](https://doi.org/10.1145/3580835) | mmWave; long-lie confirmation; 100% precision, 98.8% recall, zero false alarms | ABS |
| T3.45 | GR-Fall — Meng et al. | Proc. ACM IMWUT 2025 | [10.1145/3749471](https://doi.org/10.1145/3749471) | Alarm only when post-fall gait shows difficulty | ABS |
| T3.46 | Ma et al. (IR-UWB radar) | Sensors 2020 | [10.3390/s20041105](https://doi.org/10.3390/s20041105) | 95% sensitivity / 92.6% specificity; >90% in a new room | ABS |
| T3.47 | Su, Ho, Rantz, Skubic | JAISE 2018 | [10.3233/AIS-170469](https://doi.org/10.3233/AIS-170469) | Fusing 2 radars cut FAR 10× | ABS |
| T3.48 | Kianoush et al.; Paulikas & Paulikienė | IEEE IoT-J 2016; Sensors 2026 | [10.1109/JIOT.2016.2624800](https://doi.org/10.1109/JIOT.2016.2624800); [10.3390/s26061930](https://doi.org/10.3390/s26061930) | RSS HMM fall detection; BLE channel sounding | ABS |
| T3.49 | Wrist accelerometer/barometer fall detector | IEEE EMBC 2017 | [10.1109/EMBC.2017.8036778](https://doi.org/10.1109/EMBC.2017.8036778) | False alarms on sit-to-floor: a hard negative across sensing modalities | ABS |

### T4. ESP32 tools, systems and public datasets

| ID | Paper / resource | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T4.1 | ESP32-CSI-Tool — Hernandez & Bulut | IEEE WoWMoM 2020, pp. 277-286 | [10.1109/WoWMoM49955.2020.00056](https://doi.org/10.1109/WoWMoM49955.2020.00056); [README](https://github.com/StevenMHernandez/ESP32-CSI-Tool) | Moving the RX gave +29.4% / +28.2%. README: use baud ≥921600, FreeRTOS tick 1000 Hz, ESP-IDF v4.3, no clock sync (SETTIME / host timestamps) | ABS; README FT |
| T4.2 | Wi-ESP — Atif, Muralidharan, Ko, Yoo | J. Comput. Des. Eng. 7(5), 2020 | [10.1093/jcde/qwaa048](https://doi.org/10.1093/jcde/qwaa048) | ESP32 CSI collection tool | ABS |
| T4.3 | Espressif esp-csi | GitHub | [README](https://github.com/espressif/esp-csi) | A dedicated transmitter is the most reliable traffic mode. Documents NO_MEM errors and corrupted serial lines | FT (repo) |
| T4.4 | ESP-IDF `wifi_csi_info_t`, v4.3 / v5.2 / v5.5 | ESP-IDF source | [v4.3](https://raw.githubusercontent.com/espressif/esp-idf/v4.3/components/esp_wifi/include/esp_wifi_types.h); [v5.2](https://raw.githubusercontent.com/espressif/esp-idf/v5.2/components/esp_wifi/include/esp_wifi_types.h); [v5.5](https://raw.githubusercontent.com/espressif/esp-idf/v5.5/components/esp_wifi/include/local/esp_wifi_types_native.h) | v4.3: `mac` and `first_word_invalid`, no header. v5.2: adds `hdr`, `payload`, `dmac`. v5.5: adds `rx_seq` | FT |
| T4.5 | CSI-BERT — Zhao, Chen, Meng, Li, Li, Zhu | arXiv 2403.12400, 2024 (INFOCOM workshop per search) | [arXiv](https://arxiv.org/html/2403.12400v1) | ESP32-S3 with router ping at 100 Hz: 14.51% average loss (1-70% per second). Recovery MSE 1.73 vs 2.83 for linear interpolation | FT |
| T4.6 | WiGesture / WiFall / WiCount (RS2002 group) | Hugging Face / GitHub | [WiFall card](https://huggingface.co/datasets/RS2002/WiFall/blob/refs%2Fpr%2F1/README.md) | ESP32-S3 + router, 100 Hz, 52 subcarriers; WiFall: 10 people, 5 actions + fall. Card text partly LLM-generated. The WiCount card says "multiple receivers", which conflicts with the papers | DS |
| T4.7 | Strohmayer & Kampel (Wallhack1.8k) | IEEE ICIP 2024 | [10.1109/ICIP51287.2024.10647666](https://ar5iv.labs.arxiv.org/html/2401.01388) | ESP32-S3, ESP-NOW at a fixed 100 Hz. NLOS: biquad antenna 92.0±3.5% vs built-in PIFA 86.8±4.7%. [Zenodo 8188999](https://zenodo.org/record/8188999) | FT |
| T4.8 | Strohmayer & Kampel | ICVS 2023 (LNCS) | [10.1007/978-3-031-44137-0_4](https://doi.org/10.1007/978-3-031-44137-0_4) | Through-wall presence with a biquad antenna; Zenodo 8021099 | SNIP |
| T4.9 | Lim, ..., Leow | Data in Brief 2024 | [10.1016/j.dib.2024.111101](https://doi.org/10.1016/j.dib.2024.111101) | Router TX + ESP32 RX (SafeSense topology), 52 subcarriers + RSSI, 10 activities, 8 subjects, LOS/NLOS | ABS |
| T4.10 | WiFi-Wild — Elkelany, Ross, McKeever | IEEE Data Descriptions 2026 | [10.1109/IEEEDATA.2026.3681245](https://doi.org/10.1109/IEEEDATA.2026.3681245); [Zenodo](https://zenodo.org/records/18431784) | ESP32-CSI-Tool. F1 −21% for furniture or TX/RX moves; −7-12% for interference or clothing. Access on request | ABS |
| T4.11 | Armenta-Garcia, Gonzalez-Navarro, Caro-Gutierrez, Garcia-Reyes | Sensors 25(19):6220, 2025 | [PMC12526573](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526573/) | ESP32, 50 pps, 22 subjects incl. falls. Deep model 59.91% → 97.55% with EMD augmentation. At 230,400 baud: ASCII ~30 pps vs binary ~85 pps | FT |
| T4.12 | SHD-HAR — Wong, Wong, Peh, Tan | Data in Brief 55:110673, 2024 | [10.1016/j.dib.2024.110673](https://pmc.ncbi.nlm.nih.gov/articles/PMC11268129) | 4 ESP32-S3 (2 pairs), 40 MHz, 166 subcarriers, 60-90 pkt/s. Micro-gestures 47-55% | FT |
| T4.13 | Matey-Sanz et al. | IPIN 2023 | [10.1109/IPIN57070.2023.10332214](https://doi.org/10.1109/IPIN57070.2023.10332214) | 52% relative accuracy loss when tested 10 min after training | ABS |
| T4.14 | Pivaro Monteiro et al. (Classroom-HAR) | IEEE Access 2026 | [10.1109/ACCESS.2026.3693596](https://doi.org/10.1109/ACCESS.2026.3693596) | 5 subjects. Near-perfect on random splits; degrades under drift and LOSO. Data released | ABS |
| T4.15 | OpenCSI — Khamaisi & Rodrigues | IEEE LCN 2026 | [10.1109/LCN67947.2026.11660791](https://doi.org/10.1109/LCN67947.2026.11660791) | S3/C3/C6 mesh. Per-link quiet-baseline z-score: F1 up to 0.99 zero-shot vs 0.87 | ABS |
| T4.16 | Choi et al. (Wi-CaL) | IEEE Access 2022 | [10.1109/ACCESS.2022.3155812](https://doi.org/10.1109/ACCESS.2022.3155812) | Multi-link counting; leave-one-session-out 91.4% / 98.1% | ABS |
| T4.17 | Kabir et al. (CSI-DeepNet) | IEEE Access 2022 | [10.1109/ACCESS.2022.3217910](https://doi.org/10.1109/ACCESS.2022.3217910) | 20 gestures, 10 subjects, 96.31%; split not stated | ABS |
| T4.18 | STAR — Liu K. et al. | Sensors 26(12):3692, 2026 (DOI?); arXiv 2510.26148 | [arXiv](https://arxiv.org/abs/2510.26148) | ESP32-S3 + RV1126 NPU; 93.52% over 7 classes; 6× NPU speed-up | ABS |
| T4.19 | maxVSTAR — Liu | arXiv 2510.26146, 2025 | [arXiv](https://arxiv.org/abs/2510.26146) | 93.52% → 49.14% on uncalibrated hardware; one vision-guided adaptation cycle gets back to 81.51% | ABS |
| T4.20 | ESPARGOS | arXiv 2408.16377 / 2502.09405 | [arXiv](https://arxiv.org/html/2502.09405v1) | Phase-coherent ESP32 arrays need a shared clock; receivers miss frames for unknown reasons | FT |
| T4.21 | RuView ADR-012; ESP-PPB | GitHub | [ADR-012](https://github.com/ruvnet/RuView/blob/main/docs/adr/ADR-012-esp32-csi-sensor-mesh.md); [ESP-PPB](https://github.com/jonathanmuller/esp-ppb) | 20-50 ppm crystal drift → fuse features, not phase; FTM + VCTCXO phase lock | NPR |
| T4.22 | ESP32-C5 multi-TX dataset | IEEE DataPort | [10.21227/m754-py34](https://doi.org/10.21227/m754-py34) | 4 TX / 1 RX; activities and object pick-up | ABS |
| T4.23 | Wicaksono (ESP32-C6, 802.11ax) | Zenodo 2026 | [10.5281/zenodo.21467951](https://doi.org/10.5281/zenodo.21467951) | 256 subcarriers (242 active); people counting | ABS |
| T4.24 | Other ESP32 sensing papers (respiration, occupancy, identification, gait) | various | Alzaabi 2024 [10.1109/ACCESS.2024.3349700](https://doi.org/10.1109/ACCESS.2024.3349700); Burimas 2024 [10.3390/app14156458](https://doi.org/10.3390/app14156458); Khan 2023 [10.1109/ACCESS.2023.3316508](https://doi.org/10.1109/ACCESS.2023.3316508); Natarajan 2022-23 [10.1109/TIM.2023.3272374](https://doi.org/10.1109/TIM.2023.3272374), [10.1109/JIOT.2023.3308219](https://doi.org/10.1109/JIOT.2023.3308219), [10.1109/LES.2022.3214333](https://doi.org/10.1109/LES.2022.3214333); Gaiba 2024 [10.1109/CCNC51664.2024.10454655](https://doi.org/10.1109/CCNC51664.2024.10454655) (95% for 2 users, 74% for 3); Ghosian Moghaddam [10.1109/SAS58821.2023.10254010](https://doi.org/10.1109/SAS58821.2023.10254010), [10.1109/IEEEDATA.2026.3698410](https://doi.org/10.1109/IEEEDATA.2026.3698410); Zhou EIECT 2023 [10.1109/EIECT60552.2023.10441805](https://doi.org/10.1109/EIECT60552.2023.10441805); Tong ICCE 2024 [10.1109/ICCE62051.2024.10634669](https://doi.org/10.1109/ICCE62051.2024.10634669); Abdurrahman ICIC 2025 [10.1109/ICIC68054.2025.11309497](https://doi.org/10.1109/ICIC68054.2025.11309497); Aman PECCII 2026 [10.1109/PECCII70991.2026.11661852](https://doi.org/10.1109/PECCII70991.2026.11661852); Custance arXiv [2601.02177](https://arxiv.org/abs/2601.02177) (multi-person gait 39-56%); Zeeshan COMSNETS 2022 [10.1109/COMSNETS53615.2022.9668391](https://doi.org/10.1109/COMSNETS53615.2022.9668391); Dubal [10.17632/zycty7ksky](https://doi.org/10.17632/zycty7ksky); Chaudhari [10.6084/m9.figshare.34809009](https://doi.org/10.6084/m9.figshare.34809009) | Mostly random or unstated splits; 74-99% | ABS |
| T4.25 | ESP32 items seen only in snippets | various | ESP-Fi HAR, Ad Hoc Netw. 2026 [10.1016/j.adhoc.2026.104192](https://doi.org/10.1016/j.adhoc.2026.104192); PulseFi [arXiv 2510.24744](https://arxiv.org/abs/2510.24744); SA-WiSense [arXiv 2507.17623](https://arxiv.org/html/2507.17623v2); Atlantis SASI-ITE 2025 ESP32 LSTM (about 93.5% 5-fold; questionable references) [PDF](https://www.atlantis-press.com/article/126021002.pdf); Thai symposium 2026 (fall confused with empty room) [PDF](https://symposium2026.pcshstrg.ac.th/uploads/projects/83/paper_pdf_20260716_162028.pdf) | Do not cite numbers | SNIP |
| T4.26 | Datasets that are *not* ESP32 | — | MM-Fi (Atheros); UT-HAR (Intel 5300); NTU-Fi (Atheros); Widar3.0 (Intel 5300); EHUCount (lab equipment) | Avoid mislabelling these as ESP32 data | SNIP / earlier round |

### T5. Multi-receiver, multi-link and multi-antenna fusion

| ID | Paper | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T5.1 | Koupai, Bocus, Santos-Rodriguez, Piechocki, McConville | arXiv 2209.03765, 2022 | [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.03765) | OPERAnet: 1 RX 88.3% → 2 RX 92.1% (random split). 15 fused features score below 7 | FT |
| T5.2 | OPERAnet — Bocus, Li et al. | Scientific Data 2022 | [PMC9349197](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9349197/) | 2 CSI RX + passive radar + UWB + Kinect, 6 participants | SNIP |
| T5.3 | Mansouri Nejad et al. (Geometry Over Depth) | IEEE WiMob 2026 | [10.1109/WiMob70978.2026.11777920](https://doi.org/10.1109/WiMob70978.2026.11777920) | 2 ESP32 nodes, 3 subjects: best single 75.56% → fused 80.0% | ABS |
| T5.4 | stef4k MSc thesis | GitHub | [repo](https://github.com/stef4k/thesis-wifi-human-activity-recognition) | ESP32-S3, 2 links. Single well-placed link within 3-4 macro-F1 of fusion: "collect with two, deploy with one" | NPR |
| T5.5 | Lee, Ahn, Choi (Wi-Sensing) | Adv. Eng. Informatics 2020 | [10.1016/j.aei.2020.101147](https://doi.org/10.1016/j.aei.2020.101147) | Early fusion of per-RX STFT images; >96%; no single-RX baseline | ABS |
| T5.6 | Multiple receivers + hybrid CNN-LSTM | ACM DFHS'19 | [10.1145/3360773.3360878](https://doi.org/10.1145/3360773.3360878) | Early fusion; >95% | ABS |
| T5.7 | Multi-receiver CNN with decision fusion | IEEE CITS 2022 | [10.1109/CITS55221.2022.9832983](https://doi.org/10.1109/CITS55221.2022.9832983) | Late fusion; "better performance", no numbers | ABS |
| T5.8 | DeepMV — Xue et al. | Proc. ACM IMWUT 4(1), 2020 | [10.1145/3380980](https://doi.org/10.1145/3380980) | Quality-weighted attention over views; numbers not retrieved | ABS |
| T5.9 | CREWS — Chen, Zhou, Huang, Li | arXiv 2605.30356, 2026 | [arXiv](https://arxiv.org/abs/2605.30356) | ≤2.2-point degradation at 50% node dropout | ABS |
| T5.10 | Guo, Zhuang, Mao, Ho | arXiv 2410.11400, 2024 | [arXiv](https://arxiv.org/abs/2410.11400) | Passenger counting >94%; "up to 17.5% over a single pair" (SNIP) | ABS; SNIP |
| T5.11 | Exposing the CSI — Cominelli, Gringoli, Restuccia | IEEE PerCom 2023 | [ar5iv 2302.00992](https://ar5iv.labs.arxiv.org/html/2302.00992) | More antennas did not help cross-subject accuracy; cross-environment near random | FT |
| T5.12 | WiTraj | IEEE TMC 2021 | [10.1109/TMC.2021.3133114](https://doi.org/10.1109/TMC.2021.3133114) | Selects the best receiver views | ABS |
| T5.13 | Person-in-WiFi 3D — Yan et al. | CVPR 2024 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CSI_CVPR_2024_paper.html) | 1 TX + 3 RX; receiver ablation not retrieved | SNIP |
| T5.14 | Ristiana et al. | ICISS 2026 | [10.1109/ICISS71466.2026.11715356](https://doi.org/10.1109/ICISS71466.2026.11715356) | 2 ESP32-S3, NLOS, confidence-weighted late fusion; no 1-vs-2 comparison | ABS |
| T5.15 | Vejju et al. | CONIT 2026 | [10.1109/CONIT69683.2026.11620721](https://doi.org/10.1109/CONIT69683.2026.11620721) | Dual ESP32 presence; "up to +8.0 pp" (baseline ambiguous) | ABS |
| T5.16 | Sirisilla et al. | IEEE APSCON 2025 | [10.1109/APSCON63569.2025.11144180](https://doi.org/10.1109/APSCON63569.2025.11144180) | SHD-HAR 2-view fusion 84.5%; 72.4 ms on Raspberry Pi 5 | ABS |
| T5.17 | Wi-Gitation | MDPI Data 9(1):9 | [10.3390/data9010009](https://doi.org/10.3390/data9010009) | F1 ≥0.95 mixed split vs up to 0.21 lower leave-one-out | ABS |
| T5.18 | "Same Signal, Different Story" | IEEE IoT-J 2026 | [10.1109/JIOT.2026.3667804](https://doi.org/10.1109/JIOT.2026.3667804) | Gain alignment recovers up to 75% of cross-receiver loss | ABS |
| T5.19 | MultiSenseX | MDPI AI 6(1):6, 2025 | [10.3390/ai6010006](https://doi.org/10.3390/ai6010006) | Multi-view Transformer; no numbers | ABS |
| T5.20 | Khamaisi & Rodrigues, "Less is More" | arXiv 2602.10823, 2026 | [arXiv](https://arxiv.org/html/2602.10823v3) | 9 ESP32-C3 nodes, 72 links. All links (AUC 0.489-0.501) below the best single link (0.535-0.543) | FT |
| T5.21 | ReWiS — Bahadori, Ashdown, Restuccia | arXiv 2201.00869, 2022 | [arXiv](https://arxiv.org/abs/2201.00869) | ~40% over single-antenna approaches; "38% macro-diversity" (SNIP) | ABS; SNIP |
| T5.22 | Bauchspiess, Mane, Fürst-Walter, Becker | IEEE CYBER 2024 | [10.1109/CYBER63482.2024.10749623](https://doi.org/10.1109/CYBER63482.2024.10749623) | 8 ESP32 nodes over ESP-NOW, majority vote, 74% | ABS |
| T5.23 | 4-ESP32 edge array | ResearchGate preprint 2025 (venue?) | [link](https://www.researchgate.net/publication/395428434_Edge-Enabled_Human_Activity_Recognition_Using_Hybrid_Deep_Learning_and_Multi-Sensor_Wi-Fi_CSI_Arrays) | About 96% at about 70 ms; a 3×3 layout was slightly worse | ABS |
| T5.24 | Lim, Oh, Kim, Toh | Sensors 23(16):7292, 2023 | [10.3390/s23167292](https://pmc.ncbi.nlm.nih.gov/articles/PMC10459961/) | Classifier score fusion, *not* receiver fusion | FT |
| T5.25 | Fresnel-zone theory: Wital; Wang et al.; dynamic Fresnel model | arXiv 2305.14490; UbiComp 2016; Proc. ACM IMWUT 2023 | [Wital](https://arxiv.org/pdf/2305.14490); [10.1145/2971648.2971744](https://www.doi.org/10.1145/2971648.2971744); [IMWUT 2023](https://researchportal.ip-paris.fr/en/publications/towards-a-dynamic-fresnel-zone-model-to-wifi-based-human-activity/) | Motion along a zone boundary is invisible to that link: the rationale for a second receiver at 60-90° | ABS |
| T5.26 | ISAC primer; GaitSense | arXiv 2601.12980; ACM TOSN | [primer](https://arxiv.org/pdf/2601.12980); [GaitSense](https://tns.thss.tsinghua.edu.cn/widar3.0/data/TOSN_GaitSense_paper.pdf) | CSI ratio and BVP need coherent antennas or ≥3 links | Earlier round |

### T6. Augmentation, synthetic data and class imbalance

| ID | Paper | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T6.1 | RFBoost — Hou & Wu | Proc. ACM IMWUT 8(2), Art. 58, 2024 | [10.1145/3659620](https://arxiv.org/html/2410.07230v1) | Average +5.4%, but **cross-person only +0.87%**. On FallDar data: +2.7% detection rate, −2.6% FAR (protocol unspecified). Spectrogram rotation is harmful | FT |
| T6.2 | Strohmayer & Kampel | AIAI 2024, IFIP AICT 711:42-56 | [10.1007/978-3-031-63211-2_4](https://arxiv.org/html/2401.00964v1) | ESP32-S3, 100 Hz, 52 subcarriers, 3 classes. Combined augmentation 36.4%→58.9% cross-scenario; amplitude scaling −3.4 to +8.7 points. No subject split; gains often within run-to-run spread | FT |
| T6.3 | RF-Diffusion — Chi, Yang, Wu, Xu et al. | ACM MobiCom 2024 | [10.1145/3636534.3649348](https://arxiv.org/html/2404.09140v1) | Cross-domain +4.7% / +11.5%. Accuracy peaks, then falls, as synthetic volume grows | FT |
| T6.4 | Zhang, Wu, Wei et al. (Dense-LSTM) | IEEE IoT-J 8(6):4628-4641, 2021 | [10.1109/JIOT.2020.3026732](https://researchportal.northumbria.ac.uk/en/publications/data-augmentation-and-dense-lstm-for-human-activity-recognition-u/) | 8 CSI transforms; "up to 34.6%" gain | ABS |
| T6.5 | Zinys, van Berlo, Meratnia | Sensors 21(23):7852, 2021 | [10.3390/s21237852](https://doi.org/10.3390/s21237852) | CycleGAN translation toward the left-out domain; numbers not retrieved | ABS |
| T6.6 | CM-GR — Zhai, Dai, Jin, Qin, Zeng | Sensors 25(18):5783, 2025 | [10.3390/s25185783](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473887/) | Skeleton-conditioned synthesis: cross-subject +9.29 points on average (up to +10.26). Needs the target user's video plus unlabelled CSI | FT |
| T6.7 | DiRA — Wang, Zhao, Du et al. | arXiv 2502.12622, 2025 | [arXiv](https://arxiv.org/html/2502.12622v1) | Diffusion augmentation for counting: 0.41 → 0.70 | FT |
| T6.8 | CLAR — Xiao, Han, Yang, Hou, Shi, Chetty | arXiv 2408.05567 (IEEE IoT-J, accepted) | [arXiv](https://arxiv.org/abs/2408.05567) | Diffusion + contrastive augmentation; no numbers in abstract | ABS |
| T6.9 | SimHumalator — Vishwakarma et al. | IEEE AESM 2022; RadarConf 2021 | [arXiv 2103.01677](https://arxiv.org/pdf/2103.01677) | Mocap-simulated Wi-Fi radar; +8% when added to real data | SNIP |
| T6.10 | WiFi-JEPA | arXiv 2607.11064, 2026 | [arXiv](https://arxiv.org/html/2607.11064v1) | Ray-traced primitives for pre-training; 97.1 → 97.2 | SNIP |
| T6.11 | Vid2Doppler; Midas; RF-Genesis; mmWEAVER | CHI 2021; IMWUT 2023; SenSys 2023; arXiv 2512.11894 | [10.1145/3411764.3445138](https://karan-ahuja.com/vid2dop.html); [10.1145/3580872](https://github.com/zihanjames/midas); [Sim2Radar](https://arxiv.org/html/2602.13314v3); [mmWEAVER](https://arxiv.org/pdf/2512.11894) | Video/text → mmWave synthesis; none targets Wi-Fi CSI amplitude | ABS/SNIP |
| T6.12 | Um, Pfister, Pichler et al. | ACM ICMI 2017 | [arXiv 1706.00527](https://arxiv.org/pdf/1706.00527) | IMU: 77.54% → 86.88%. Jitter and scaling failed because intensity carries label information | ABS |
| T6.13 | Iwana & Uchida | PLOS ONE 16(7):e0254841, 2021 | [PMC8282049](https://pmc.ncbi.nlm.nih.gov/articles/PMC8282049) | 128 datasets. Time warping hurt MLP (t = −5.64) and LSTM-FCN (t = −4.66); window warping and slicing recommended | FT |
| T6.14 | IMUTube — Kwon et al. | Proc. ACM IMWUT 4(3), 2020 | [10.1145/3411841](https://arxiv.org/abs/2006.05675) | Virtual IMU data from video | ABS |
| T6.15 | Leng, Iyer, Plötz | arXiv 2506.07612, 2025 | [arXiv](https://arxiv.org/html/2506.07612v1) | LOSO: classical augmentation mixed (71.42 → 62.34 F1 in one cell); large gains only at 10% data | FT |
| T6.16 | Oishi, Birch, Roggen, Lago | arXiv 2508.13284, 2025 | [arXiv](https://arxiv.org/abs/2508.13284) | Physics-simulated augmentation: +3.7 points macro-F1; up to 60% fewer subjects needed | ABS |
| T6.17 | Tu, Lin, Liu, Chan | Sensors 25(7):2168, 2025 | [10.3390/s25072168](https://pmc.ncbi.nlm.nih.gov/articles/PMC11991046/) | Wrist falls, leave-one-group-out, full data: baseline 83.87 F1; transforms 84.14, SMOTE 84.44, diffusion 86.00 (precision fell) | FT |
| T6.18 | Debnath, Alamgeer, Kabir, Ngu | Sensors 25(15):4639, 2025 | [10.3390/s25154639](https://api.openalex.org/works/doi:10.3390/s25154639) | Diffusion-generated falls: offline F1 +7-10% | ABS |
| T6.19 | Alamgeer, Souissi, Ngu | Sensors 25(16):5144, 2025 | [arXiv 2505.04660](https://arxiv.org/html/2505.04660v1) | Synthetic falls hurt every method on SmartFallMM wrist data (0.740 → 0.578-0.710) | FT |
| T6.20 | CDL-Fall — Zhang, Li, Wang | Sensors 21(19):6511, 2021 | [10.3390/s21196511](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512051/) | Specificity: class weights 74.49%, SMOTE 66.88%, focal loss 66.49%, SMOTE + threshold 89.98%. Random window split | FT |
| T6.21 | SA-GAN — Soleimani & Nazerfard | Neurocomputing 426:26-34, 2021 | [arXiv 1903.12489](https://arxiv.org/abs/1903.12489v1) | Subject-adaptor GAN; beat state of the art in >66% of experiments | ABS |
| T6.22 | SDMix — Lu, Wang, Chen, Pan, Hu, Qin | Proc. ACM IMWUT 2022 | [10.1145/3534589](https://arxiv.org/html/2206.06629) | Cross-person: ERM 70.26, vanilla mixup 71.21, SDMix 77.21. Mixup hurt cross-dataset | FT |
| T6.23 | Kang, Kim & Toh | Sensors 25(5):1547, 2025 | [10.3390/s25051547](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902660/) | 70.86% → 98.92% on ARIL's predefined split (not subject-wise) | FT (earlier round) |
| T6.24 | SynCheck — Gong, Liang, Gao, Xu | ACM MobiSys 2025 | [arXiv 2506.23174](https://arxiv.org/abs/2506.23174) | Unfiltered synthetic CSI −13.4%; quality filtering +4.3% | ABS |
| T6.25 | Snippet-only augmentation items | arXiv 2607.01882; arXiv 2603.22313; ESWA 2024 | [additive noise](https://arxiv.org/pdf/2607.01882); [focal vs BCE](https://arxiv.org/html/2603.22313); [MaP-SGAN](https://www.sciencedirect.com/science/article/abs/pii/S0957417424009497) | Additive noise poorly matches real CSI; focal→BCE recall 98.9% → 95.1% | SNIP |
| T6.26 | Missing-cell analogues | PMC 2022; arXiv 2411.14070 | [compositional splits](https://pmc.ncbi.nlm.nih.gov/articles/PMC9009598/); [Gajanin et al.](https://arxiv.org/html/2411.14070v1) | Held-out-combination splits and label-skew work; no method for a fully missing subject × class cell | ABS |

### T7. Irregular sampling, packet loss and rate mismatch

| ID | Paper | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T7.1 | CSI-BERT2 — Zhao, Meng, Lyu, Li, Li, Zhu | arXiv 2412.06861 v4; page header says IEEE TMC (acceptance?) | [arXiv](https://arxiv.org/html/2412.06861v4) | WiFall fall detection on data with packet loss: CSI-BERT2 94.79% vs CNN 52.99% (75.78% with linear interpolation). Train 100 Hz / test 50 Hz: 94.32% vs 76.13%. Recovery lowered the best model from 99.48% to 89-92% | FT |
| T7.2 | SRV-NN — Yin, Zhang, Shen, Cotton | arXiv 2605.08308, 2026 | [arXiv](https://arxiv.org/html/2605.08308) | Real traffic runs at 1-352 Hz. Rate mismatch: 78.1% → 28.0% and 93.6% → 55.1%. Rate augmentation brings the average to 0.96 | FT |
| T7.3 | Zakeri, Nguyen, Juntti (age of samples) | arXiv 2606.31690, 2026 | [arXiv](https://arxiv.org/html/2606.31690v1) | At 1% sensing budget: CSI + sample age 80.7% vs CSI only 72.7% (HumanID) | FT |
| T7.4 | Kayano, Nishio, Yoda, Hirai, Adachi | arXiv 2603.11858, 2026 | [arXiv](https://arxiv.org/html/2603.11858v2) | 8 ESP32 stations (ESP32-CSI-Tool, ch 6, 20 MHz). Station-wise masking for missing stations | FT |
| T7.5 | Klein Brinke, Chiumento, Havinga | PerCom Workshops 2023 | [10.1109/PerComWorkshops56833.2023.10150403](https://research.utwente.nl/en/publications/channel-state-information-for-human-activity-recognition-with-low/) | F1 >85% at lower or interpolated rates | ABS |
| T7.6 | Sharma, Mishra, Jha, Seneviratne | IEEE INFOCOM Workshops 2024 | [10.1109/INFOCOMWKSHPS61880.2024.10620749](https://doi.org/10.1109/INFOCOMWKSHPS61880.2024.10620749) | Packet timing is more uniform at higher rates | ABS |
| T7.7 | Tanno, Maekawa, Hara | MobiQuitous 2021 / LNICST 2022 | [10.1007/978-3-030-94822-1_31](https://doi.org/10.1007/978-3-030-94822-1_31) | Adaptive sampling-rate selection (RL, per snippet) | SNIP |
| T7.8 | WiImg2.0 — Zheng, Yang, Xiong, Liu, Ma | IEEE TMC 2024 | [10.1109/TMC.2024.3374046](https://doi.org/10.1109/TMC.2024.3374046) | At 25 pkt/s, GAN inpainting: gestures 59.1% → 86.7%, activities 65.9% → 96.4% | ABS |
| T7.9 | RoboFiSense | arXiv 2312.15345 | [arXiv](https://arxiv.org/pdf/2312.15345) | At 10 Hz, only 20-40 packets fall within each movement | SNIP |
| T7.10 | Liu et al. (wearable sampling rate) | IEEE TCDS 2021 | [10.1109/TCDS.2021.3116228](https://doi.org/10.1109/TCDS.2021.3116228) | Lower sampling rates degrade fall detection | ABS |
| T7.11 | Related rows elsewhere | — | See T1.26 (FuseFi: missing-rate and SCV definitions), T4.5 (CSI-BERT), T9.1 (Hernandez: interpolation unnecessary at a constant 100 Hz), T5.9 (CREWS) | — | — |

### T8. Environment and static-object robustness

| ID | Paper | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T8.1 | Zhang, Wang, Wen, Zhu | Scientific Reports 2024 | [10.1038/s41598-024-54077-x](https://pmc.ncbi.nlm.nih.gov/articles/PMC10864388/) | Presence detection on Intel 5300. Moving furniture without retraining cost about 1.8-2.1 TPR points | FT |
| T8.2 | Monjur, Liu, Xu et al. | arXiv 2402.09452, 2024 | [arXiv](https://arxiv.org/html/2402.09452v1) | 8 environments, 16 people. Unseen environment: 82% → 42% (63% with GRL). Unseen people: 12-20% drop. Temporal drift minimal | FT |
| T8.3 | Kindt et al. | TechRxiv 2023 | [10.36227/techrxiv.22284421.v1](https://www.authorea.com/users/689826/articles/681364-a-dataset-on-csi-based-activity-recognition-in-real-world-environments) | Moving small objects is the main cause of poor generalization; 70 GB dataset | SNIP |
| T8.4 | Liu, Wang, Jiang, Chen | arXiv 2002.05770, 2020 | [arXiv](https://arxiv.org/abs/2002.05770) | Presence detection "near perfect"; no numbers | ABS |
| T8.5 | One-class SVM on empty-room data (TU Berlin) | 2023 (authors?) | [PDF](https://bits.informatik.hu-berlin.de/~zubow/pd_ocsvm_ew_2023.pdf) | ">97% in most cases" | SNIP |
| T8.6 | TCD-FERN | arXiv 2304.13107 | [arXiv](https://arxiv.org/pdf/2304.13107) | Multi-room presence >97% | SNIP |
| T8.7 | CRONOS | arXiv 2211.10354 | [arXiv](https://arxiv.org/pdf/2211.10354) | A motionless occupant is hard to tell from an empty room | SNIP |
| T8.8 | Finger-gesture noise removal | arXiv 2106.00857 | [arXiv](https://arxiv.org/pdf/2106.00857) | Noise removal helps under furniture changes; no numbers | SNIP |

### T9. Edge, tiered, MEC and standards

| ID | Paper / standard | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T9.1 | Hernandez & Bulut, "WiFi Sensing on the Edge" | IEEE COMST 25(1), 2023 (online 2022) | [10.1109/COMST.2022.3209144](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf) | Only 11 of 176 papers report inference rate. ESP32 on-board inference runs at 145→1.5 Hz (float) and 492.1→1.1 Hz (INT8). Currents: 58 mA idle; +17 mA for inference; 110-121 mA active RX; about 7 mA deep sleep. A constant 100 Hz stream needs no interpolation. Z-scoring cut accuracy from 87.36% to 70.39%. A 3-link home deployment gave single links of 58.5%, 71.2% and 49.9%, against an oracle of 89.1% | FT |
| T9.2 | Zhu, Hu, Wu, Wang, Wang, Liu (Origin Research) | arXiv 2506.04322, 2025 | [arXiv](https://arxiv.org/html/2506.04322v1) | 280 devices in 15 homes. Event-driven upload of compact ACF features: −61.72% data rate and −99.72% daily data. Non-human false alarms fell from 63.1% to 8.4%. Devices qualify only below 10% packet loss | FT |
| T9.3 | EfficientFi — Yang, Chen, Zou, Wang, Xu, Xie | IEEE IoT-J 9(15):13086-13095, 2022 | [10.1109/JIOT.2021.3139958](https://arxiv.org/pdf/2204.04138) | Edge-to-cloud CSI compression: >1,700× less traffic at >98% HAR accuracy | ABS/SNIP |
| T9.4 | RSCNet — Barahimi et al. | IEEE ICC 2024 | [arXiv 2402.04888](https://arxiv.org/abs/2402.04888) | Dynamic CSI compression; 97.4% accuracy | ABS |
| T9.5 | WECAR — Li et al. | arXiv 2503.07669, 2025 | [arXiv](https://arxiv.org/abs/2503.07669) | ESP32 runs inference, Jetson edge runs training; no numbers | ABS |
| T9.6 | Snippet-only edge items | various | SwinFi [arXiv 2405.03957](https://arxiv.org/pdf/2405.03957); Chakraborty, Ad Hoc Netw. 2025 [PDF](https://www.cse.iitm.ac.in/~ayon/files/chakraborty_wisdom_adhoc25.pdf); industrial edge, Comput. Netw. 2026 [link](https://www.sciencedirect.com/science/article/abs/pii/S1389128626004056); early exit [arXiv 2507.20623](https://arxiv.org/html/2507.20623v1) | Industrial edge: 21 ms at <30% CPU. Early exit: energy −10.4% to −37.8%, sometimes with higher latency | SNIP |
| T9.7 | Koo, Liang, Mishra, Seneviratne | Energies 19(2):573, 2026 | [10.3390/en19020573](https://www.mdpi.com/1996-1073/19/2/573) | Proposes early-exit or cascaded heads and lower sampling rates in static scenes; no measurements | SNIP |
| T9.8 | IEEE 802.11bf-2025 | IEEE SA | [standard page](https://standards.ieee.org/ieee/1641/10365/) | Approved 28 May 2025, published 26 September 2025 | ABS (search summary) |
| T9.9 | Du et al., 802.11bf overview | arXiv 2310.17661 (journal?) | [arXiv](https://arxiv.org/pdf/2310.17661) | Uncompressed CSI is the only sub-7 GHz feedback type; taxonomy of edge inference | FT |
| T9.10 | Meneghello, Chen, Cordeiro, Restuccia | IEEE Commun. Mag. 2023 | [10.1109/MCOM.001.2200806](https://api.openalex.org/works?filter=doi:10.1109/MCOM.001.2200806) | 802.11bf ISAC position paper | Metadata only |

### T10. Surveys and position papers

| ID | Paper | Venue, year | DOI / link | Open problems it names | Verif. |
|---|---|---|---|---|---|
| T10.1 | Ma, Zhou, Wang | ACM CSUR 52(3), 2019 | [10.1145/3310194](https://dl.acm.org/doi/10.1145/3310194) | Robustness, privacy, sensing–network coexistence | SNIP |
| T10.2 | Tan, Ren, Yang, Chen | IEEE IoT-J 2022 | [10.1109/JIOT.2022.3164569](https://arxiv.org/pdf/2111.07038) | Multi-user sensing needs >1000 pkt/s; multipath change; data scarcity | FT |
| T10.3 | Chen, Zhou, Lin | ACM CSUR 55(11), 2023 | [10.1145/3570325](https://dl.acm.org/doi/10.1145/3570325) | Cross-domain taxonomy covering >60 papers | ABS |
| T10.4 | Miao, Huang, Lu, Ohtsuki, Gui, Sari | ACM CSUR 2024/25 | [10.1145/3705893](https://dl.acm.org/doi/10.1145/3705893) | Generalization, real-time processing | SNIP |
| T10.5 | Ahmad, Ullah, Choi | IEEE OJ-COMS 2024 | [10.1109/OJCOMS.2024.3411529](https://ieeexplore.ieee.org/abstract/document/10552143) | Multi-user sensing, human-in-the-loop | SNIP |
| T10.6 | Wang et al., generalizability survey | IEEE COMST 2026 | [arXiv 2503.08008](https://arxiv.org/html/2503.08008v3) | Human body diversity barrier; silos ("Tower of Babel"); synthetic data | SNIP / earlier round |
| T10.7 | Radwan, Yildirim, Hasanzadeh, Tabassum, Valaee | IEEE COMST 2025 (DOI 10.1109/COMST.2025.3586212 per arXiv) | [arXiv 2506.12052](https://arxiv.org/abs/2506.12052) | Self-supervised learning; dataset scarcity | ABS |
| T10.8 | Schumann, Li, Grzegorzek | iWOAR 2023 | [10.1145/3615834.3615841](https://doi.org/10.1145/3615834.3615841) | Single-antenna AAL needs realistic ADL tests; multi-link work rare but promising | ABS |
| T10.9 | Chua et al. | 2026 | [10.47852/bonviewaia62027517](https://doi.org/10.47852/bonviewaia62027517) | >20% drops in unseen buildings; multi-user interference −21% to −46% | ABS |
| T10.10 | Fall and modality reviews: Liu 2026; Hu 2024 (radar); Ishaq 2026 (PRISMA) | various | [10.54097/k0fmnk74](https://doi.org/10.54097/k0fmnk74); [10.1109/MRA.2024.3352851](https://doi.org/10.1109/MRA.2024.3352851); [10.3390/app16041929](https://doi.org/10.3390/app16041929) | Context only | ABS |
| T10.11 | Snippet-only surveys | various | [Wireless fall review (ResearchGate)](https://www.researchgate.net/publication/402631285_Wireless_Fall_Detection_An_Integrative_Investigation_of_Deep_Learning_Innovations_Challenges_and_Future_Directions); [CMC 2025](https://www.sciencedirect.com/org/science/article/pii/S1546221825010380); [SoK CSI biometrics](https://arxiv.org/pdf/2511.11381) | Fall/non-fall similarity, limited real fall data | SNIP |

### T11. Evaluation methodology

| ID | Paper | Venue, year | DOI / link | What it shows | Verif. |
|---|---|---|---|---|---|
| T11.1 | Kapoor & Narayanan | Patterns 4(9):100804, 2023 | [10.1016/j.patter.2023.100804](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856) | Leakage in ≥294 papers; "model info sheets" | FT |
| T11.2 | Saeb, Lonini, Jayaraman, Mohr, Kording | GigaScience 6(5), 2017 | [10.1093/gigascience/gix019](https://papers.ecosyste.ms/papers/10.1093%2Fgigascience%2Fgix019) | Use subject-wise cross-validation | ABS |
| T11.3 | Little et al. | GigaScience 2017 | [10.1093/gigascience/gix020](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441396) | The CV scheme must match the deployment question | ABS |
| T11.4 | Hammerla & Plötz | UbiComp 2015 | [10.1145/2750858.2807551](https://unpaywall.org/10.1145%2F2750858.2807551) | Overlapping windows bias random CV | ABS |
| T11.5 | Dehghani, Sarbishei, Glatard, Shihab | Sensors 19(22):5026, 2019 | [10.3390/s19225026](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891351/) | Overlap gains vanish under subject CV; the ~10% / 16% inflation figures (arXiv 1904.02666) are SNIP | FT; SNIP |
| T11.6 | Tello, Degeler, Lazovik | arXiv 2310.11950 | [arXiv](https://arxiv.org/abs/2310.11950) | Overestimation from random-split practice is still widespread | ABS |
| T11.7 | Gholamiangonabadi, Kiselov, Grolinger | IEEE Access 8:133982-133994, 2020 | [10.1109/ACCESS.2020.3010715](https://openaccess.library.uitm.edu.my/Record/doaj-9bea9b3dbf2c4991ab92d75c8c999e3c) | Same model: 99.85% under 10-fold CV vs 85.1% under LOSO | ABS |
| T11.8 | Varoquaux | NeuroImage 180:68-77, 2018 | [10.1016/j.neuroimage.2017.06.061](https://arxiv.org/abs/1706.07581) | Error bars of about ±10% at n = 100 | ABS |
| T11.9 | Blagus & Lusa; Santos et al.; Demircioğlu; Vandewiele et al. | BMC Bioinf. 2015; IEEE CIM 2018; Sci. Rep. 2024; arXiv 2020 | [10.1186/s12859-015-0784-9](https://www.doi.org/10.1186/s12859-015-0784-9); [Santos](https://student.dei.uc.pt/~miriams/pdf-files/Santos18-IEEE-CIM.pdf); [10.1038/s41598-024-62585-z](https://www.nature.com/articles/s41598-024-62585-z.pdf); [arXiv 2001.06296](https://arxiv.org/pdf/2001.06296) | Oversampling before splitting inflates results | ABS |
| T11.10 | Lones, "How to avoid ML pitfalls" | arXiv 2108.02497 | [arXiv](https://arxiv.org/pdf/2108.02497) | Augmenting before the split is a critical error | SNIP |
| T11.11 | Dietterich | Neural Computation 10(7):1895-1923, 1998 | 10.1162/089976698300017197 | McNemar's test for comparing classifiers. Cited from memory and **not fetched** | Unverified |

## Conclusion

**The survey changes what counts as the bottleneck.** In the published ESP32 literature, packet provenance is invisible. Papers report packet rates at best and never say which transmitters produced the CSI. Our data show that this one unreported variable can push a five-subject fall detector to chance. Until both sides control source and rate, SafeSense-5G cannot meaningfully compare itself with hardware-matched published numbers.

**The useful cues on this hardware are slow ones.** These receivers are 20 MHz and single-antenna, and they sit behind an ASCII serial link at about 70 Hz. On that setup the cues that survive are posture level change, burst envelope and stillness, not Doppler spectra. The fall logic should rest on those cues. Doppler-style features become an upgrade once a fixed-rate transmitter is in place.

**Person independence at five subjects is a calibrate-at-install problem.** The remaining sit↔stand confusion hinges on which static CSI state means which posture for a given person in a given spot. Only that person's few labelled trials can supply it.

**Several of the most publishable results are likely to be nulls.** Augmentation that only moves the operating point, DANN that ties ERM, and a second receiver that adds a few points are all likely outcomes. They become informative only because the project runs the controls the literature usually skips: λ = 0, leave-one-cell-out, and an inner-CV-selected single receiver.

## Prioritized next steps

### Pre-registration rules

Run every experiment under the same LOSO folds, at trial level, with these rules:

- Fit every learned or tuned element inside the fold. That includes scalers, thresholds, class weights, augmentation and style ratios.
- Use 5 seeds.
- Report Wilson intervals for sensitivity and false-alarm rate.
- Compute cluster-bootstrap intervals by resampling trials within subjects.
- Use McNemar tests on pooled trial-level decisions.
- Report subject 3's fold as false alarms only.

The success criteria below are our proposed pre-registration thresholds, not literature values.

### Experiment plan

| Priority | Experiment | Protocol | Success criterion | Decision it drives |
|---|---|---|---|---|
| P0-1 | Freeze the v2 data layer with a per-trial provenance audit | Per trial and per RX: dominant-source share, sig_mode mix, cross- vs within-source jump statistic, missing rate (MR) on a 100 Hz grid, sampling coefficient of variation (SCV), max gap, RSSI | Every retained trial ≥75% dominant source. Within-source jump statistic ≈2.1 throughout. Empty-room motion energy falls from ~0.5 to ≤0.25 of the human-trial median | Freeze v2; flag or drop failing trials |
| P0-2 | Confound probes | Subject-ID probe; within-subject block-ID probe; RSSI-only and day-only classifiers for empty room | Report all three. If RSSI alone separates empty room at AUC ≥0.95, move empty room out of headline macro-F1 and label it session-confounded. If block ID is predictable well above chance, add leave-one-block-out CV | Scopes every later claim |
| P0-3 | v2 LOSO baseline vs original pipeline | Random forest / HistGB on v2 features, trial level | Pooled fall-vs-ADL AUC ≥0.75 with bootstrap lower bound >0.65. AUC >0.5 for each of subjects 1, 2, 4 and 5. McNemar vs original pipeline p < 0.05 | Establishes the baseline for all later ablations |
| P0-4 | Recalibrate Tier 1 / Tier 2 thresholds in-fold, with a post-fall latch | Thresholds from training-fold ADL quantiles. Log the fraction of windows exiting at each tier and the falls lost at each gate | Tier 1 passes ≥77 of 79 falls. Fast-path receives ≤5% of ADL windows (now 99.96%) | Makes the tier architecture real |
| P1-5 | Event-centred two-stage fall detector | Label-free peak detection; 2-5 s segment sweep; burst-shape features; 2-4 s stillness confirmation | Sensitivity at ≤8% ADL FAR rises from 46.8% to ≥60%, with paired bootstrap Δ excluding 0. Full ROC published | Fall-logic design |
| P1-6 | 1-vs-2 RX matrix on v2 | RX1; RX2; inner-CV-selected single; early fusion; late fusion (mean probability); OR/AND fall fusion. Per-RX quiet-baseline normalization in all conditions. Per-RX MR/SCV logged | Pre-registered H1: best fusion beats the selected single receiver, with ΔAUC bootstrap interval excluding 0 and Holm-adjusted McNemar p < 0.05. Otherwise report the null with its interval | Deploy one or two receivers |
| P1-7 | k-shot calibration curve | k = 1, 2, 3 trials per class from the held-out subject, removed from its test set. Compare RF refit (up-weighted), nearest-class-mean, and label-free quiet-baseline normalization (transductive, reported separately) | k = 1 gives ≥ +10 points macro-F1. Sit↔stand confusion down ≥50% by k = 3 | Calibrate-at-install workflow |
| P1-8 | Sit/stand temporal structure; hand detectability | HMM or state machine over within-trial window posteriors. Hand-vs-empty AUC on v2 | Sit↔stand confusion halved zero-shot. If the hand-vs-empty AUC interval includes 0.5, declare hand below the detection floor and merge it into low-motion ADL | Class definition |
| P2-9 | Leave-one-cell-out test of subject-style augmentation | For j ∈ {1, 2, 4, 5}: remove j's falls, keep j's non-falls, train with and without j-style augmentation | Adopt only if mean fall sensitivity at fixed in-fold FAR improves ≥5 points with interval excluding 0. Otherwise publish as a negative result | Subject 3 strategy |
| P2-10 | One-class ADL novelty arm | Model trained on non-fall trials only, including subject 3 | Fall-vs-ADL AUC within 0.05 of supervised v2 | Candidate Tier-2 model that needs no falls |
| P2-11 | DANN ablation | Same MLP: λ = 0 vs DANN vs CORAL vs class-conditional. Empty-room and fall windows excluded from the domain loss. Nested selection; 5-10 seeds | Adopt only if Δ macro-F1 exceeds the across-seed SD and fall recall does not drop. Otherwise ship ERM plus calibration | Server-tier classifier |
| P2-12 | Sampling robustness | Decimate to 50 / 35 / 25 Hz; Bernoulli packet drop 10-50%; masked vs interpolated vs gap-feature inputs | Locate the knee where AUC loss exceeds 0.03. Publish the per-RX MR/SCV table | Firmware requirements |
| P2-13 | Physics-style features | Amplitude-STFT band fractions below the effective Nyquist; DTW distance to a training-fold fall template | ΔAUC interval excludes 0 | Keep or drop |
| P3-14 | Firmware and traffic | Binary output, ≥1.5 Mbaud or SD logging. On-device MAC and sig_mode filter. ESP-IDF ≥5.5 (`rx_seq`, `hdr`). Fixed-rate ESP-NOW or ping transmitter at 100 Hz | Dominant-source share ≥95%; MR ≤0.1; 99th-percentile gap <30 ms; RX1/RX2 frame overlap reported | Unlocks Doppler-band features and frame pairing |
| P3-15 | New recordings | About 20 real mattress falls for subject 3. Randomized, interleaved trial order. Empty room recorded within sessions. 10-20 each of sit-to-floor, lie-down and squat/pick-up. 3-5 more subjects of varied body size | Session probe AUC ≈0.5. FAR reported separately on hard negatives | Turns the dataset into a citable benchmark |
| P3-16 | Tier measurement | Per-tier exit fraction, alert latency, compute and uplink bytes, ESP32 current | Report all four. Alert latency ≤6 s. Uplink reduction stated against raw CSI (about 149 MB/h per link) | Edge contribution |
| P3-17 | External validation on WiFall (ESP32-S3) | Cross-dataset fall vs non-fall, both directions | Report only | Context, not claims |

### Sequencing and stop rules

The order matters. P0 items are prerequisites, because every P1-P2 result computed on the contaminated pipeline is uninterpretable. Two stop rules apply:

- **If P0-3 fails** (pooled AUC below 0.7), fix the data layer before running any model ablation.
- **If P3-14 changes the data distribution**, re-run P0-3 before comparing against old results.

## Research contributions this project can credibly claim (and what it must not claim)

### Contributions within reach

Several gaps found in this survey are within reach of a 2-ESP32, 5-subject, one-room project. Each claim below needs to be backed by the listed experiment and framed as stated.

| Credible claim | Evidence required | Novelty basis in this survey |
|---|---|---|
| A documented and quantified multi-transmitter contamination problem in ESP32-CSI-Tool router-TX captures, with its downstream effects (fake empty-room motion, tier mis-routing, chance-level LOSO) and a filter that restores discrimination | P0-1, P0-3 (AUC 0.48 → ~0.76, confirmed with intervals) | No surveyed ESP32 paper reports per-source composition. FuseFi harmonizes packet types on Nexmon for a different goal ([FuseFi](https://arxiv.org/html/2512.22143)). Frame it as "we document", not "first" |
| Per-receiver irregular-sampling statistics (MR, SCV, gap distribution, serial bottleneck) for a two-receiver ESP32 dataset, with rate- and loss-robustness curves | P0-1, P2-12 | No surveyed ESP32 paper publishes gap distributions; FuseFi provides the metric definitions |
| A capacity-controlled 1-vs-2 ESP32 fall-detection comparison under LOSO, including a null result if that is the outcome | P1-6 | No such ablation found. The closest is a 3-subject HAR study ([WiMob 2026](https://doi.org/10.1109/WiMob70978.2026.11777920)) |
| An honest ESP32 LOSO fall baseline with sensitivity, FAR and intervals, scoped to the negatives actually recorded | P0-3, P1-5 | No surveyed ESP32 fall paper reports LOSO with both sensitivity and FAR |
| A leave-one-cell-out protocol for a missing subject × class cell, plus a negative or positive result for style augmentation | P2-9 | No paper found studies a missing cell |
| A measured tier-cascade accounting: exit fractions, falls lost per gate, alert latency, compute and uplink saved | P0-4, P3-16 | Only 11 of 176 papers report inference rate ([Hernandez & Bulut](https://www.ib.edu.ar/fisica-exp/images/e/e0/COMST22_WiFi_Sensing_Survey.pdf)). No CSI early-exit or per-tier energy paper found |
| A DANN-on-handcrafted-features ablation with a λ = 0 control under label shift | P2-11 | No Wi-Fi paper applies DANN to handcrafted feature vectors |
| A released dataset with per-packet source, sig_mode and timestamp metadata, furniture labels and the evaluation protocol | P0-1, P3-15 | Data scarcity and benchmark fragmentation rank second among survey-named open problems |

### Claims the project must not make

The project **must not claim** any of the following:

- **Person independence.** Say "cross-subject generalization across 5 subjects in one room" instead.
- **Cross-environment robustness.** There is one room. The furniture test covers only empty-room trials, which are session-confounded.
- **The ≥95% sensitivity / ≤3% FAR targets**, unless they are met with intervals on realistic negatives.
- **Performance on real or elderly falls.** All falls were simulated on a mattress.
- **FAR on hard negatives**, until sit-to-floor, lie-down and pick-up trials exist.
- **Any fall result for subject 3.**
- **Synthetic falls as data.**
- **A DANN benefit** without the λ = 0 control.
- **A significant second-receiver gain** from a single run.
- **Hand-gesture recognition**, if hand stays at the noise floor.
- **Battery-life savings from Tier 1.** The radio draws 110-121 mA whether or not inference runs.
- **"First early-exit for CSI"**, which has not been checked in IEEE Xplore or the ACM DL.
- **802.11bf compatibility.** The receivers are 802.11n.
- **5G MEC latency or performance.** No 5G link was measured. Unless Tier 3 runs over a measured 3GPP 5G path, call it an "edge server (MEC-style)".

## Addendum: v2 pipeline results (our measurements, 2026-11-20, after this report was drafted)

**Setup.** The v2 pipeline:

- `merge_dual_rx.py --source-filter dominant --drop-subcarriers 0,27-37`, giving 104 channels;
- `csi_preprocess.py --dc-mode median`;
- `feature_extraction.py --extended` with 2 s windows and a 0.25 s stride, giving 30 features for dual-RX and 21 for RX1-only;
- `train_classifier.py --label-mode event`.

All results are trial-level LOSO over subjects 1-5, with the empty-room trials spread across folds. Each condition is one run with one seed. The intervals are a bootstrap over trials, not the cluster bootstrap the plan above calls for.

| Condition | Model | Accuracy | Macro-F1 | Fall-vs-ADL AUC [95% CI] | Fall sens. / ADL FAR (argmax + τ = 0.5 override) |
|---|---|---|---|---|---|
| Original pipeline (22 features, 0.5 s windows) | RF | 33.6% | 0.30 | 0.48 [0.40-0.54] | 3.8% / 9.2% |
| v2, dual-RX | RF | 42.6% | 0.42 | 0.73 [0.67-0.79] | 45.6% / 10.0% |
| v2, dual-RX | HistGB | 42.2% | 0.43 | 0.69 [0.62-0.75] | 57.0% / 24.7% |
| v2, RX1 only | RF | 46.4% | 0.46 | 0.76 [0.69-0.82] | 48.1% / 8.5% |
| v2, RX1 only | HistGB | 44.0% | 0.46 | 0.76 [0.69-0.82] | 68.4% / 27.0% |
| v2, dual-RX, 50 quality-flagged trials excluded | RF | 42.9% | 0.43 | 0.72 [0.66-0.80] | 42.4% / 9.3% |
| v2, dual-RX + style augmentation (training folds only) | RF | 42.6% | 0.43 | 0.74 [0.68-0.80] | 49.4% / 15.0% |
| v2, dual-RX + generic and style augmentation | RF | 40.9% | 0.41 | 0.75 [0.68-0.81] | 59.5% / 18.8% |

Trial-grouped 5-fold accuracy on v2 is 54.2% for dual-RX RF and 54.6% for RX1 RF.

**Reading.**

1. The v2 data layer reproduces the proof of concept: AUC rises from 0.48 to about 0.73-0.76. P0-3's criterion is met by RX1 RF (0.755, lower bound 0.69) but not quite by dual-RX RF (0.732). The per-subject AUCs and the McNemar test still need to be computed.
2. Dropping the flagged trials changes nothing.
3. Augmentation adds about +0.01 AUC, well inside the intervals. It mainly shifts the operating point toward "fall", which matches the literature summarized above.
4. **1-vs-2 receivers is unresolved.** With v2 window features, RX1 alone matches or beats dual-RX, which reverses the proof of concept's trial-level features (0.704 vs 0.762). This is exactly why P1-6 needs RX2-only, an inner-CV-selected single receiver, late fusion, several seeds, and paired tests before anything is claimed.

### Addendum 2: Antigravity's answers, P1-6, P2-9 and P0-4 (our measurements, 2026-11-20)

**What Antigravity confirmed (AGENT_BRIDGE.md, Batch 2).**

- The MAC addresses: 1C:61:B4:93:EF:FA is the router BSSID; 30:F6:EF:47:8A:3D and 10:F6:0A:42:E4:E7 are Laptops A and B; 04:D5:90:… are neighbouring APs sending legacy frames.
- RX2's true MAC is 1C:C3:AB:B4:A2:48. The KT value was a DHCP-lease string.
- Subjects did not keep to the timeline: subjects 1 and 2 moved at the GO beep, subject 4 about 1 s later, and subject 5 around 5.2 s.
- The mattress was present only for falls, and the chair only for sit/stand.
- The laptops were moved between recording days.
- The Tier thresholds were placeholders.

Antigravity's explanation of the `mac` field is internally inconsistent. It says the field is the receiver address, but it also says laptop-MAC frames were transmitted by the laptop. Own-MAC HT frames are router→ESP32 under either reading. We therefore added a third data layer, **v3**: `--source-filter own`, meaning only HT frames whose mac is the receiver's own MAC.

**v3 vs v2 (dual-RX, RF, LOSO).** v3 gives higher accuracy (49.7% vs 42.6%; macro-F1 0.48 vs 0.42) and a similar fall AUC (0.71 [0.63-0.77] vs 0.73 [0.67-0.79]). RX1-only v3 reaches macro-F1 0.50. We recommend v3 as the default, because it is physically unambiguous and better on accuracy, with fall AUC inside the interval.

**P1-6: 1 vs 2 ESP32 receivers.**

The setup: 5 seeds, RF with 200 trees, LOSO, trial level, identical trials and folds in every condition. McNemar tests are Holm-adjusted. Δ intervals come from a cluster bootstrap over trials within subjects.

| Data layer | RX1 | RX2 | SELECTED (inner-CV single) | EARLY fusion | LATE fusion |
|---|---|---|---|---|---|
| v2 macro-F1 / fall AUC | 0.463 / 0.751 | 0.398 / 0.709 | 0.463 / 0.751 | 0.426 / 0.729 | 0.452 / 0.761 |
| v3 macro-F1 / fall AUC | **0.502** / 0.734 | 0.407 / 0.683 | 0.502 / 0.734 | 0.479 / 0.726 | 0.481 / 0.735 |

- **H1 (fusion beats the selected single receiver) is not supported** on either data layer.
- EARLY vs SELECTED gives ΔF1 [−0.073, −0.004] (v2) and [−0.058, +0.012] (v3).
- LATE vs SELECTED gives ΔF1 [−0.049, +0.025] (v2) and [−0.054, +0.012] (v3). All Holm-adjusted p ≥ 0.16.
- Inner CV picked RX1 in 25 of 25 fold×seed cases.
- RX1 beats RX2 significantly on v3 (McNemar p_holm = 0.001; ΔF1 [+0.047, +0.142]).
- Late fusion lowers ADL false alarms from about 6.5% to 4.4%, but only by lowering sensitivity. Its AUC is unchanged.
- **Conclusion:** with these features and this placement, a second ESP32 adds no measurable value over the better-placed single receiver. Receiver placement matters more than receiver count.
- Caveat: RX2's packet rate is lower (median 59.7 vs 64.9 Hz on v2), which confounds receiver identity with placement.

**P2-9: leave-one-cell-out test of subject-style augmentation.**

The setup: v2 features. For each j in {1, 2, 4, 5}, j's falls were hidden while j's other activities stayed in training; each s ≠ j was fully held out; 3 seeds; 16 (j, s) pairs.

| Arm | Missing-cell fall sensitivity | Unseen-subject ADL FAR | Unseen-subject fall sensitivity |
|---|---|---|---|
| NONE | 28.9% | 5.2% | 27.4% |
| GENERIC augmentation | 38.4% | 8.9% | 37.2% |
| STYLE augmentation (subject-j energy scale) | 39.3% | 9.1% | 38.1% |

- STYLE − NONE: sensitivity +10.5 points [+7.0, +14.4], FAR +3.9 points [+2.4, +5.3]. **The pre-registered rule is not met, so do not adopt.**
- STYLE ≈ GENERIC: the gain comes from extra fall examples shifting the operating point, not from matching the subject's style.
- Without augmentation, missing-cell sensitivity (28.9%) is about equal to fully-unseen sensitivity (27.4%). **The feared "subject 3 → not fall" shortcut is not observed.**

**P0-4: Tier 1/2 gates calibrated in-fold** (LOSO, v2). T1 is the 2.5% quantile of the training-fold fall-trial maximum motion; T2 is the 95% quantile of the training-fold ADL event-window burst.

| Gate metric | Windows skipped at Tier 1 | Falls lost at Tier 1 | Falls reaching fast-path | ADL trials on fast-path | Empty-room windows skipped |
|---|---|---|---|---|---|
| Scale-free ext_* | 73% | 2/79 | 36.7% | 8.8% | 57% |
| Raw energy (original metrics) | 56% | 4/79 | 8.9% | 7.8% | 83% |

Routing is real now; the placeholder thresholds sent 99.96% of windows to the fast-path. Two consequences:

- The fast-path cannot serve as the fall detector, so Tier 3 is essential.
- Tier 1 needs a session-level absolute noise floor rather than per-trial normalization.

