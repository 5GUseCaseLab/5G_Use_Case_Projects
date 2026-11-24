# SafeSense-5G should lengthen windows, not models

SafeSense-5G is already on a sound path. The literature backs tree ensembles and SVMs on physically motivated features as the primary models at 509 trials and 5 subjects. The project's biggest open risks are not in model choice. They are in four places: the **0.5 s windows**, the **10 Hz low-pass**, how **fall trials are labelled**, and whether the evaluation and the 1-vs-2-receiver comparison are honest. On small ESP32 datasets, the head-to-head comparisons that exist show random forest matching or beating 1D-CNNs and CNN-BiLSTM-attention models. Deep ESP32 models collapse without heavy augmentation (59.9% vs 97.6%). Every model family loses roughly 13-30 points when the test subject is unseen. Published fall systems almost all classify 1.5-5 s event segments. At 2.4 GHz, fall Doppler plausibly sits at about 16-48 Hz, well above the current 10 Hz cutoff. The targets of at least 95% sensitivity and at most 3% false alarms have been met only by physics-based systems on better hardware (DeFall, XFall). No published ESP32 fall study reaches both targets under a rigorous protocol, so the team should treat them as stretch goals and report confidence intervals. With 79 falls, the 95% CI around 95% sensitivity spans about 88-98%. Copying subject 2's falls into subject 3's slot is not acceptable. It mislabels where the data came from, and it plants exact duplicates of test trials into training. The fix is to report subject 3's fold as "no falls, false-alarm rate only" and to record real falls for subject 3. Comparing one receiver against two is a genuine gap in the literature: no paper found runs a 1-vs-2 ESP32 ablation for falls under leave-one-subject-out (LOSO). A carefully controlled version would be a novel contribution.

## Tree ensembles on physical features match deep nets at 509 trials

The 509 trials are the true sample size, not the ~20,000 overlapping windows. That one fact should anchor the model choice. The ~39 windows cut from each 10 s trial share samples and context, so a high-capacity model sees far less independent evidence than the window count suggests. The few head-to-head comparisons on ESP32 data point the same way. An open-source ESP32 benchmark (37 participants, 20 classes, 100 Hz, 2 s windows; **not peer-reviewed**) used subject-disjoint splits and compared RF and SVM on 468 handcrafted statistics against a 1D-CNN and a CNN-BiLSTM-attention on raw windows. **RF was best in every setting**: window macro-F1 of 20.75% vs 19.65% for the 1D-CNN, recording-level majority-vote accuracy of 27.38% vs 20.83%, and leave-one-location-out macro-F1 of 6.78% vs 6.04% ([gladiator-13 ESP32 HAR benchmark, GitHub](https://github.com/gladiator-13/ESP32-WiFi-CSI-Human-Activity-Recognition)). The absolute numbers are poor because a 20-class problem is hard. What matters for SafeSense-5G is the ordering, and that recording-level voting helped every model.

The deep-learning evidence on small ESP32 sets is a warning, not an endorsement. Armenta-Garcia et al. trained an LSTM-DenseNet121 on 1,198 ESP32 training samples covering five activities including falls. It reached only **59.91% without augmentation and 97.55% only after EMD-based synthetic augmentation inflated the training set to 11,513 samples**. The paper trained no RF/SVM baseline, and it does not say whether the split was subject-wise or whether augmentation touched only training data ([Armenta-Garcia et al., Sensors 25(19):6220, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526573/)). A 2026 ESP32 TinyML study found that the best model's accuracy dropped by a **mean of 29 points under leave-one-user-out** (range 63-96%), and fell from 84.1% to 51.7% cross-subject on the 35-user CSI-Bench. The repository states the study appeared in IEEE Access, but its peer-review status was not independently verified ([Shahbaz et al., esp32-csi-har-frontier](https://github.com/ahmadshahbazzz/esp32-csi-har-frontier)). The SenseFi benchmark authors state outright that deep models "confront the overfitting issue, especially for cross-domain scenarios". ViT fell to 76.84% on the 546-sample NTU-Fi Human-ID set ([SenseFi, Patterns 4(3):100703, 2023](https://arxiv.org/html/2207.07859v3)). None of SenseFi's datasets is ESP32. UT-HAR is Intel 5300 and NTU-Fi is Atheros, both with 3 antennas and 30-114 subcarriers. SenseFi's 92-99.7% accuracies, from random 8:2 splits, are therefore upper bounds that SafeSense-5G should not benchmark itself against.

The strongest counter-evidence deserves a fair hearing. In Yousefi et al.'s foundational 2017 comparison, an LSTM on raw amplitude clearly beat RF. Per-class recall read from the confusion matrices was about **0.91 for the LSTM vs 0.58 for RF (fall 0.94 vs 0.60)**. However, the RF there used generic STFT bins rather than physically meaningful features. The figures were derived from partly garbled matrices and are **approximate**. The often-quoted 90.5% / 73% / 64% overall accuracies were **not verified**. The train/test protocol is also unstated ([Yousefi et al., IEEE Communications Magazine 2017](https://ar5iv.labs.arxiv.org/html/1708.07129)). A small-sample CSI paper reports a CNN with augmentation about 5.47 points above SVM and RF, but that figure is **snippet-only and its venue is unverified** ([DF-WiSLR, ResearchGate](https://www.researchgate.net/publication/354615635_Small_CSI_Samples-Based_Activity_Recognition_A_Deep_Learning_Approach_Using_Multidimensional_Features)). On the closest analogue to this project, five subjects with two single-antenna ESP32s and seven activities including falls, Ye et al. found that their best deep metric-ensemble model averaged about 83.9% under LOSO, against about 77% for a CNN. These means were computed from the paper's per-subject table and are not stated by the paper. Even that model let fall accuracy sink to **3.82% in one fold**, through fall-vs-pick-up confusion ([Ye et al., Bioengineering 11(11):1124, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591964)). A deep model can beat the classical stack here. It will not do so by a margin that the 5-fold LOSO variance can resolve.

The practical recommendation is to keep the planned four, used in specific roles:

- **HistGradientBoosting** is the primary model. It has native class weighting, is insensitive to feature scale and is fast enough for nested CV.
- **Random forest** is the robust second.
- **SVM-RBF** needs scaling fitted inside each fold.
- The **MLP** should be small. On 22 features it has thousands of parameters, not SenseFi's 23-175 M raw-CSI MLP ([SenseFi](https://arxiv.org/html/2207.07859v3)).

The team should add a logistic-regression floor so that any gain from model complexity is visible. If a learned-feature comparison is wanted, a ROCKET-style random-kernel model plus a Ridge classifier is the cheapest one. LiteHAR showed this design is competitive with deep models without training convolution weights, although its abstract gives no numbers ([Salehinejad & Valaee, LiteHAR, ICASSP 2022](https://arxiv.org/abs/2201.09310)). A small 1D-CNN belongs in the study only as a secondary arm, on windows of 2 s or longer, with a shared-weight per-receiver encoder and augmentation applied only inside training folds. For fall detection specifically, the lesson of DeFall and XFall is that representation beats classifier. Both reach at least 95% detection because they feed speed and acceleration profiles, not raw CSI, into the decision stage ([DeFall, IEEE IoT-J 2021](https://www.semanticscholar.org/paper/0decb43be5171fcf69b8983844fd4449743b5ab4); [XFall, IEEE JSAC 2024](https://api.openalex.org/works?search=XFall%20domain%20adaptive%20fall%20detection&per_page=2)). Engineering effort spent on event-level kinematic features will pay more than effort spent on architectures. If on-device deployment ever matters, RF and decision trees run end-to-end in about 1-2 ms on a classic ESP32. LSTMs do not convert to TFLite-Micro at all ([Shahbaz et al.](https://github.com/ahmadshahbazzz/esp32-csi-har-frontier)). No head-to-head study of gradient boosting against deep models for CSI HAR was found, so HistGB's ranking is something the project must establish itself.

## Unseen subjects will cost 13-20 points, so report it per subject

The literature is unambiguous that subject-mixed evaluation flatters CSI models, and the size of the penalty depends on how controlled the data are. SafeSense-5G has a fixed room, fixed TX/RX geometry and scripted actions. That puts it closest to Ye et al.'s ESP32 study, where same-person accuracy of 91-97% fell to roughly **72-89% per held-out subject**, a drop of about 13-20 points ([Ye et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591964)). Less controlled datasets collapse far harder:

| Setting | In-domain / mixed | Subject-disjoint | Source |
|---|---|---|---|
| ESP32, 5 subjects, 7 acts incl. fall (LOSO, means computed by notes) | 91-97% | ~77% CNN, ~84% best | [Ye et al., Bioengineering 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591964) |
| Nexmon CSI-HAR, 3 subjects, 2D-CNN | 92.2% | 66.4% | [Varga, Sensors 24(24):8201, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679234/) |
| CommCSI-HAR (ESP32-class), FuseFi | 96.9% | 41.6% | [Dong et al., FuseFi, IEEE IoT-J 2026](https://arxiv.org/html/2512.22143) |
| WiMANS, identity-dependent macro-F1 | 80.38 | 32.61 | [Bayik et al., arXiv 2604.16572, 2026](https://arxiv.org/html/2604.16572v1) |
| ImgFi on WiAR, precision | 99.0% | 23.4% | [Varga, Sensors 24(10):3159, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11124867/) |

Two notes on these rows. A further claim, that RF on SHARP-80 fell from 0.954 to 0.181 under LOSO, was seen **only in a search result and is unverified** ([ResearchGate](https://www.researchgate.net/publication/417208606_Reliability_Assessment_of_Wi-Fi_CSI-Based_Activity_Recognition_Under_Environmental_Shift_and_Structured_Measurement_Loss)). Elkelany et al. argue that subject bias leads the literature to overstate accuracy and that LOSO should be the standard, but **their numbers could not be extracted** ([Elkelany et al., AICS 2024](https://ceur-ws.org/Vol-3910/aics2024_p78.pdf)). A realistic prior for SafeSense-5G is trial-grouped CV in the mid-to-high 90s and **LOSO roughly 75-88%**, with a wide spread across folds. Hand gesture will be the weakest class: ESP32 micro-gesture accuracy was only 47-55% in SHD-HAR ([Wong et al., Data in Brief 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11268129)). This range is an inference from comparable studies, not a measured result for this dataset.

Physics limits which "domain-invariant" tricks this hardware can use. CSI ratio, inter-antenna phase difference and body-velocity profiles (BVP) all depend on phase-coherent antennas sharing one oscillator, or on three or more well-placed links ([ISAC primer, arXiv 2601.12980](https://arxiv.org/pdf/2601.12980); [GaitSense, ACM TOSN](https://tns.thss.tsinghua.edu.cn/widar3.0/data/TOSN_GaitSense_paper.pdf)). A stock ESP32-WROOM-32 has one antenna, and two separate boards have independent crystals. Cross-RX phase is therefore meaningless; phase-coherent ESP32 arrays such as ESPARGOS need a shared clock and a phase reference ([ESPARGOS, arXiv 2408.16377](https://arxiv.org/pdf/2408.16377)). Three things remain feasible:

- Per-receiver amplitude spectrograms, a "DFS-like" motion-speed profile computed from the STFT of the leading PCA components.
- Temporal phase differencing within a single receiver, after linear sanitization.
- Agreement features between the two receivers' motion envelopes.

The team's per-RX phase sanitization is defensible. Amplitude-derived features should carry the weight.

The cheapest levers against subject shift are measurement and normalization, not learned domain adaptation. Adversarial domain adaptation (EI, MobiCom 2018) and meta-learning (RF-Net, OneFi) were built for deep encoders and many source domains. With four training subjects per fold, a domain discriminator has four "domains" to learn from ([EI, MobiCom 2018](https://dblp.org/rec/conf/mobicom/JiangMMYWYXSMKX18.html); [OneFi, SenSys 2021](https://dl.acm.org/doi/10.1145/3485730.3485936)). The following steps are untested on this pipeline and should each be validated by LOSO ablation:

- **Static-baseline normalization:** divide amplitude per subcarrier by its median over the trial's 0-2 s still segment.
- **Scale-free features:** prefer band-energy fractions, normalized variance and dominant frequency over absolute amplitude, which encodes body size.
- **Subject-ID probe:** train a classifier to predict subject identity from the 22 features. If it scores high, the features carry identity, and the most identity-laden ones are candidates for removal ([Bayik et al.](https://arxiv.org/html/2604.16572v1)).
- **Per-subject standardization:** z-scoring with the test subject's unlabelled windows is transductive and must be reported as a separate condition.

The most measurable lever is few-shot calibration. FuseFi raised LOSO accuracy from **41.6% to 91.3% by fitting only a linear head on 5 labelled windows per activity** from the held-out subject, and to 94.7% with 10 ([FuseFi](https://arxiv.org/html/2512.22143)). SafeSense-5G can reproduce this cheaply. Add k = 1, 3 or 5 trials per class from the held-out subject to training, refit, and plot the recovery curve against zero-shot LOSO.

Honest reporting matters as much as the models. Five subjects are enough to *report* cross-subject generalization; Ye et al. used exactly five. They are not enough to *claim* person independence. Ye et al. themselves call for more subjects and note their cohort's narrowness: all male, 165-175 cm, 50-68 kg ([Ye et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591964)). Large cross-domain claims rest on cohorts like CrossSense's 100 participants ([Chen et al., ACM CSUR 2023](https://dl.acm.org/doi/10.1145/3570325)). SafeSense-5G should report:

- per-subject accuracy, macro-F1 and fall recall, plus mean ± SD and the worst subject;
- per-fold confusion matrices;
- the trial-grouped-CV minus LOSO gap, stated as the "subject-shift cost";
- the subject-ID probe result.

The headline wording should be "cross-subject generalization across 5 subjects in one room", not "person-independent". Empty_room contains no person, so it cannot be a LOSO fold. If it stays in the per-fold averages it inflates LOSO accuracy. Its 30 trials should be distributed across folds stratified by furniture configuration, reported separately, and additionally tested leave-one-furniture-configuration-out.

## Falls last 1-2 s, and their Doppler sits above the 10 Hz cutoff

The current 0.5 s window is shorter than any window found in the CSI fall literature. That literature classifies event segments of 1.5-5 s:

- XFall reports its best F1 at a 1,500 ms window. The window detail comes from the XFall PDF and was **not verified from full text** ([XFall](https://tns.thss.tsinghua.edu.cn/~guoxuan/assets/pdf/Paper-XFall.pdf)).
- A Tsinghua Science & Technology paper notes that a fall lasts about 1 s and uses a 3 s window (**snippet-level**) ([Tsinghua Sci. & Tech.](https://www.sciopen.com/local/article_pdf/10.26599/TST.2022.9010011.pdf)).
- Chu et al. use 5 s segments, also from a **snippet, unverified** ([Chu et al., IEEE Access 2023](https://eprints.whiterose.ac.uk/id/eprint/202278/8/Deep_Learning_Based_Fall_Detection_Using_WiFi_Channel_State_Information.pdf)).
- The 2026 ESP32 proof-of-concept classifies 5 s windows ([Amritha et al., arXiv 2611.15532](https://arxiv.org/html/2611.15532)).

The one controlled ablation found is decisive in direction. In an acceleration-based CSI fall study, shortening the window from 120 to 40 samples **cut the true-positive rate from 87.49% to 73.00% and raised the false-positive rate from 8.92% to 18.42%**. Window length mattered more than packet rate ([DP-AcE, arXiv 2401.00160, 2024](https://arxiv.org/pdf/2401.00160)). A 50-sample window also gives about 2 Hz PSD resolution, so the 0.5-2 Hz band is effectively one bin. The 0.5 s window is also IMU practice imported to Wi-Fi: 0.5 s with 50% overlap is typical for 200 Hz wearables ([arXiv 2603.22313](https://arxiv.org/html/2603.22313v1)). It is not typical for CSI.

The 10 Hz Butterworth cutoff probably removes the band where falls differ most. CSI power fluctuates at roughly the rate at which reflected path length changes, divided by wavelength ([CARM, MobiCom 2015](https://old.sigmobile.org/mobicom/2015/papers/p65-wangA.pdf)). At 2.4 GHz, λ ≈ 12.5 cm. A torso or head descending at 1-3 m/s changes path length at up to about twice that speed, which puts fall energy at **roughly 16-48 Hz, against a few Hz for a slow sit-down**. This arithmetic is a physics inference from the notes, not a measured result. CARM's guidance that human motion occupies CSI components below 300 Hz comes from a **search snippet** ([CARM](https://old.sigmobile.org/mobicom/2015/papers/p65-wangA.pdf)). One construction-site study reports "serious degradation" at around 50 Hz sampling, but **its attribution was not verified** ([IJERPH 20(6):4998, 2023](https://www.mdpi.com/1660-4601/20/6/4998)). The ESP32 proof-of-concept used the same 10 Hz low-pass and missed 2 of 16 falls (87.5% recall), on fall-vs-walk only ([Amritha et al.](https://arxiv.org/html/2611.15532)). That neither validates nor refutes the cutoff.

One complication is specific to this project. The ESP32 streams arrive at 50-130 Hz and are resampled to 100 Hz. Any trial whose native rate dips to 50 Hz carries no real content above 25 Hz, whatever the resampled grid suggests. The team should do three things:

- Log the effective native packet rate per trial and per receiver.
- Run the cutoff as an ablation (10, 20 and 40 Hz, and none, i.e. ~45 Hz anti-alias only) under LOSO.
- Add 10-20 Hz and 20-40 Hz band-energy fractions and an upper-spectral-edge ("max Doppler") feature, flagging or excluding those features on trials whose native rate cannot support them.

These features are the closest single-antenna analogue of the time-frequency cues that RT-Fall and FallDeFi relied on ([RT-Fall, IEEE TMC 2017](https://researchportal.ip-paris.fr/en/publications/rt-fall-a-real-time-and-contactless-fall-detection-system-with-co/); [FallDeFi, PACM IMWUT 2018](https://api.openalex.org/works/https://doi.org/10.1145/3161183)).

Event segmentation is the norm, not an embellishment. From WiFall onward, fall systems first detect a motion burst or anomaly, then classify the segmented event ([Nguyen & Nguyen, CISS 2020](https://arxiv.org/pdf/2005.11932)). RT-Fall segments on the "sharp power profile decline" that ends a fall. DeFall matches speed and acceleration time series against templates ([RT-Fall](https://researchportal.ip-paris.fr/en/publications/rt-fall-a-real-time-and-contactless-fall-detection-system-with-co/); [DeFall](https://www.bohrium.com/paper-details/defall-environment-independent-passive-fall-detection-using-wifi/811714708896743425-2451)). SafeSense-5G should restructure along the same lines:

- Keep the 0.5 s windows as a motion-energy trigger.
- Cut an event segment of about 1 s before to 2-3 s after the energy peak.
- Compute event-level features on that segment: rise time and peak dE/dt, burst duration, upper spectral edge, the ratio of peak energy to energy over the following 1-2 s, and time from peak to stillness.
- Classify the segment.

The current labelling also needs fixing. In a fall trial, 0-2 s and roughly 6-10 s are static. Labelling all ~39 windows "fall" teaches the model that stillness is a fall. That is the confusion an ESP32-S3 project reported, where falls were mistaken for an empty room because the person lay still (**snippet-only**) ([Thai symposium 2026 project](https://symposium2026.pcshstrg.ac.th/uploads/projects/83/paper_pdf_20260716_162028.pdf)). Window labels should come from the 2-6 s action interval, or from the detected event, with static windows given a "still" label or excluded.

Post-fall stillness is a confirmation cue, not a discriminator, in this dataset. Every scripted trial ends still: sit ends seated, stand ends standing. Stillness alone therefore cannot separate a fall from a sit-down. What separates them is the shape of the burst followed by stillness. A fall decelerates abruptly into near-total quiet. A fast sit-down decelerates over a longer interval, often with small upper-body adjustments. Crouch and pick-up are followed by a recovery burst within seconds. This contrast is reasoned in the notes from RT-Fall's power-decline cue and DeFall's kinematics, not measured on ESP32 data. The deployment logic should be:

1. trigger on motion energy;
2. classify the event;
3. confirm with 2-4 s of stillness near the empty-room baseline and no recovery burst;
4. alert, with "pending" and "confirmed" states.

Expected latency is 4-6 s after impact, which is acceptable for fall alerting. The planned trial-level "fall override" (any window above threshold τ makes the trial a fall) raises sensitivity at the cost of false alarms, because a single noisy 0.5 s window from a hand gesture can trip it. The team should require several consecutive windows or an event-level score, tune τ only inside inner training folds, and publish the full trial-level sensitivity vs false-alarm curve rather than one operating point.

The targets should be read against what the field has achieved:

| System (hardware) | Detection | False-alarm rate | Setting | Source |
|---|---|---|---|---|
| WiFall, INFOCOM 2014 / TMC 2017 (Intel 5300) | 87%; 90-94% (journal) | 18%; 13-15% | same domain | [Nguyen & Nguyen 2020](https://arxiv.org/pdf/2005.11932); [Scilit](https://www.scilit.net/publications/c8829155edd304f3d09f06ffaac95391) |
| WmFall, IJDSN 2018 | 89% | 8% | average | [SAGE](https://journals.sagepub.com/doi/full/10.1177/1550147718805718) |
| FallDeFi, IMWUT 2018 | 93% → ~80% | n/a | trained room → changed room | [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1145/3161183) |
| DeFall, IoT-J 2021 (5.8 GHz, 40 MHz) | >95% | <1.5% | new environments, no retraining | [Semantic Scholar](https://www.semanticscholar.org/paper/0decb43be5171fcf69b8983844fd4449743b5ab4) |
| XFall, JSAC 2024 | 96.8% acc., 3.1% miss | 3.3% | year-long deployment | [OpenAlex](https://api.openalex.org/works?search=XFall%20domain%20adaptive%20fall%20detection&per_page=2) |
| Physics-attention Transformer, TMC 2026 (Intel 5300) | 99.5% pooled; 88.3-91.7% unseen rooms | "near-zero" (no number) | no cross-subject test | [arXiv 2605.00869](https://arxiv.org/html/2605.00869) |
| ESP32 proof-of-concept, arXiv 2026 | 87.5% recall (14/16) | 0% (13 walks) | fall vs walk only | [arXiv 2611.15532](https://arxiv.org/html/2611.15532) |

The statistical arithmetic makes the targets tighter than they look. With 79 falls, each miss costs about 1.3 points of sensitivity. Reaching 95% allows **at most 3 misses**, and even 75/79 carries a Wilson 95% interval of about **88-98%**. A per-subject 19/20 has an interval of about 76-99%. On the false-alarm side, 12 false alarms in 400 non-fall human trials is exactly 3%, and its Wilson interval runs from about 1.7% to 5.2%. That calculation is ours from the stated counts. The dataset also lacks the hardest negatives. Lie-down and pick-up were the classes most confused with falls in Ye et al. and Varga ([Ye et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591964); [Varga 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679234/)), and sit/stand was the weakest live class (86.7%) in the 2026 Transformer study ([arXiv 2605.00869](https://arxiv.org/html/2605.00869)). A ≤3% false-alarm rate measured on walk, sit, stand and hand will overstate deployment specificity unless fast sit-downs, lie-downs and pick-ups are added as non-fall trials.

## A second ESP32 must beat both single receivers, not one

The case for a second receiver is real but conditional. In Fresnel-zone terms, motion along a zone boundary of a given TX-RX pair barely changes path length, so any single link has blind directions ([Wital, arXiv 2305.14490](https://arxiv.org/pdf/2305.14490)). A receiver 60-90° away sees differently oriented ellipses, so motion that is silent on link 1 is visible on link 2. WiFall's authors attributed its false alarms to the "limited angle of observation" of one link, and added a second link to cut them ([Cardenas et al., Sensors 21(15):5121, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8347439/)). ReWiS reports that multi-receiver "macro-diversity improves the prediction accuracy by 38%" over a single receiver. That figure is **search-snippet only; the full text was not read** ([ReWiS, arXiv 2201.00869, 2022](https://arxiv.org/abs/2201.00869)).

Widar3.0 degrades slowly from 6 links to 3, then drops sharply below 3 ([Widar3.0, MobiSys 2019](https://cswu.me/papers/mobisys19_widar3_paper.pdf)). One follow-up reports below 70% accuracy with two links, a **snippet not individually verified** ([IEEE IoT-J 2021](https://huangqy7.github.io/Paper/Context-Aware_Wireless-Based_Cross-Domain_Gesture_Recognition.pdf)). Those systems reconstruct velocity geometrically, which this project cannot do, so their link-count curve does not transfer directly.

The counter-evidence is just as important. A 2026 study ran nine ESP32-C3 nodes as a full mesh (72 links) for occupancy detection. Concatenating all links (AUC 0.489-0.501) did *worse* than the single best link (0.535-0.543), and soft attention over links did not help (0.460 ± 0.070) ([Khamaisi & Rodrigues, "Less is More", arXiv 2602.10823, 2026](https://arxiv.org/html/2602.10823v3)). Those AUCs sit near chance and come from 72 links, so the "dilution effect" is a caution for 2 links, not a verdict. Antenna orientation alone moved single-link fall accuracy from 50% (vertical-vertical) to 92% (horizontal-horizontal) ([Cardenas et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8347439/)). Placement and polarization of the *single* receiver therefore deserve a mention as an alternative explanation for any 1-vs-2 gap. No paper found reports a 1-vs-2 ESP32 ablation for fall detection under LOSO. The experiment below would fill a genuine gap.

The design principle is to hold everything fixed except the information available. Build one master index of the 509 trials, with windows cut on the shared 100 Hz grid. Apply the same exclusion mask for heavily interpolated windows to *every* condition, so RX1-only is not scored on a cleaner subset. Log per-RX effective packet rate, because a systematic rate gap between receivers is itself a confound ([ESPARGOS, arXiv 2502.09405](https://arxiv.org/html/2502.09405v1)).

The current 22 features reportedly include cross-RX amplitude correlation. The feature set must therefore be redefined *per receiver*. Single-RX conditions drop the cross-RX features. Fusion conditions add them back as a separately ablated block. Fusion should happen at feature or decision level, because two ESP32s drift 20-50 ppm and share no phase reference. That drift figure comes from a **non-peer-reviewed engineering note** ([RuView ADR-012](https://github.com/ruvnet/RuView/blob/main/docs/adr/ADR-012-esp32-csi-sensor-mesh.md)).

| Priority | ID | Condition | Classical-pipeline definition | Question it answers |
|---|---|---|---|---|
| P0 | E1 | RX1-only | 22-type features from RX1 only, no cross-RX terms | Single-view baseline A |
| P0 | E2 | RX2-only | Same features from RX2 only | Single-view baseline B ("1 RX" depends on which one) |
| P0 | E3 | Early (feature-level) fusion | E1 ∥ E2 features plus cross-RX correlation/agreement block | Does a second view help when the model sees both? |
| P0 | E4 | Late fusion | Mean of E1 and E2 trial-level probabilities; variant with weight fit in inner CV | Does fusion work when views are individually competent? |
| P1 | E5 | Fall decision fusion | OR vs AND on per-RX event-level fall scores | Explicit sensitivity vs false-alarm trade-off (WiFall's two-link rationale) |
| P1 | C1 | Width control | E1 features ∥ an equal-width block of permuted or noise-perturbed RX1 features | Is any E3 gain new spatial information or just extra dimensions? |
| P1 | C2 | Cross-RX-block ablation | E3 without the cross-RX correlation block | Is the gain from the second view or from cross-RX features? |
| P1 | C3 | Honest single-RX selection | Pick RX1 or RX2 per fold by inner CV; also report test-oracle best single (labelled oracle) | The bar fusion must clear; tests whether the chosen RX is just lucky |
| P2 | E6 | Intermediate fusion (deep) | Shared-weight per-RX 1D-CNN encoder → concat/mean/attention, ≥2 s windows | Best default for learned features on small data; parameters ≈ single-RX model |
| P2 | E7 | Early fusion (deep) | 128-channel 1D-CNN; capacity-matched to E6 and to a 64-channel RX1 CNN | Deep baseline; checks dilution on raw input |
| P3 | R1 | Packet-loss robustness | Drop 10-50% of RX2 packets at test time only | Graceful degradation (late fusion expected to degrade gracefully) |
| P3 | R2 | Missing receiver | Fusion models evaluated with one RX zeroed or imputed | Operational resilience |
| P3 | R3 | Per-class complementarity | Per-class F1 for E1 vs E2; stratify by fall direction if recorded | Mechanism behind any fusion gain (Fresnel geometry) |

Every condition runs under the same LOSO folds and the same trial-grouped secondary CV. Each gets identical hyperparameter budgets, chosen by nested inner CV or fixed beforehand, and 3-5 seeds per fold. Report parameter or feature counts.

The statistics must respect five folds. A two-sided Wilcoxon signed-rank test over 5 subjects **cannot reach p < 0.05: its minimum is p = 0.0625**. The 5-subject comparison should therefore be reported as five per-subject deltas with their signs and an effect size. The paired inference should come from the following:

- McNemar's test on pooled trial-level predictions: the same test trials scored under two conditions. This is the standard recommendation from Dietterich (Neural Computation 1998), which was cited from memory and **not fetched**.
- A cluster bootstrap over trials within subjects, giving 95% intervals on the *difference* in macro-F1, fall recall and false-alarm rate.
- Holm-Bonferroni correction across the pairwise family.

The dilution paper is a recent CSI example of this style. It used paired t, Wilcoxon, Cohen's d and bootstrap intervals, and tested best-SNR link against a random link (p = 0.35) as a luck control ([arXiv 2602.10823](https://arxiv.org/html/2602.10823v3)). To keep the family small, pre-register two primary hypotheses. First, best fusion (E3 or E4) beats the inner-CV-selected single receiver (C3) on trial-level macro-F1. Second, fusion lowers the false-alarm rate at matched fall sensitivity. Everything else is exploratory.

## Copying subject 2's falls would fabricate data and leak the test set

Duplicating subject 2's fall trials and labelling them as subject 3 fails on integrity grounds and on statistical grounds at once. The integrity problem is simple: the data would state that subject 3 performed falls they never performed. Results reported "for subject 3" would then be a false statement about where the data came from, and disclosure would not make the metric meaningful.

The statistical damage is concrete. In the LOSO fold that tests subject 2, the training set would contain **bit-identical copies of the test falls**, each with about 39 identical windows, so a tree ensemble can memorise them. This is the "duplicates across train and test" failure in Kapoor & Narayanan's leakage taxonomy. They found leakage in at least 294 papers across 17 fields, and in one case study the apparent advantage of complex models disappeared once leakage was fixed ([Kapoor & Narayanan, Patterns 4(9):100804, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)). In the fold that tests "subject 3", the fall recall would measure recognition of subject 2's falls, and say nothing about a new person. That defeats the reason for LOSO, which Saeb et al. argue must "approximate the use-case" ([Saeb et al., GigaScience 6(5), 2017](https://papers.ecosyste.ms/papers/10.1093%2Fgigascience%2Fgix019)). The trial-grouped CV leaks too. If the copies receive new trial IDs, an original and its copy can land on opposite sides of a split.

The CSI literature shows how large such leakage effects are. Varga found that random rather than subject-based splitting turned a model scoring **23.4% precision on WiAR into one reporting 99.0%** ([Varga, Sensors 24(10):3159, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11124867/)). Duplication also achieves nothing that legitimate reweighting cannot. The imbalance is mild: 79 falls against 100 trials for each other activity class. Class weights computed inside each training fold deliver the same emphasis without leakage or misattribution.

The alternatives are straightforward:

1. **Keep subject 3 in LOSO for the non-fall classes.** In that fold, report fall recall as "N/A (no fall trials)" and report subject 3's false-alarm rate on their non-fall trials, which is a deployment-relevant number in its own right.
2. **Pool fall sensitivity over subjects 1, 2, 4 and 5**, and state explicitly that fall results rest on four folds.
3. **Balance with class weights** computed inside each training fold.
4. **Augment, if at all, only inside the training fold**, keeping each augmented copy's group ID equal to its source trial. Oversampling or SMOTE before the split is a documented source of overoptimism ([Blagus & Lusa, BMC Bioinformatics 2015](https://www.doi.org/10.1186/s12859-015-0784-9); [Demircioğlu, Scientific Reports 2024](https://www.nature.com/articles/s41598-024-62585-z.pdf); [Vandewiele et al., arXiv 2001.06296](https://arxiv.org/pdf/2001.06296)). SMOTE on 50%-overlapping windows mostly re-creates the same trial.
5. **Best fix: record about 20 real, safe mattress falls for subject 3.** This beats every workaround.

On augmentation, label-preserving transforms are plausible: small time shifts that keep the event in frame, mild ±10-20% time-warping, per-subcarrier amplitude scaling, noise at the measured floor, and subcarrier dropout. Their effect on falls has not been validated. Doppler or frequency-shift augmentation should be avoided, because it can turn a fast fall into a slow sit-down. Published gains are seductive but often come from loose splits. Kang et al. report 70.86% rising to 98.92% with noise, shifting and MixUp, on ARIL's predefined split, with other splits unstated ([Kang et al., Sensors 25(5):1547, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902660/)). GAN and diffusion synthesis is riskier still. Unfiltered synthetic CSI **degraded task performance by 13.4%** in one MobiSys 2025 study ([Gong et al., MobiSys 2025](https://arxiv.org/abs/2506.23174)), and a generator trained on 79 falls from four people is unlikely to add real diversity.

## Trial-level, nested, cluster-bootstrapped evaluation is the defensible protocol

The planned split design is sound. LOSO over subjects 1-5 is the headline, and StratifiedGroupKFold by trial is the secondary "seen subjects, unseen trials" estimate. The gap between them measures subject shift. The planned design falls short in three places: how windows are treated, where preprocessing is fitted, and how uncertainty is reported.

Overlapping windows from the same recording are not independent. Random CV over them is biased ([Hammerla & Plötz, UbiComp 2015](https://unpaywall.org/10.1145%2F2750858.2807551)). The accuracy gains usually attributed to overlap vanish under subject-independent CV ([Dehghani et al., Sensors 19(22):5026, 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891351/)). Dehghani et al.'s conference version puts the inflation at about 10% from subject-dependent CV, and about 16% with overlap, but those figures are **from a search summary and should be checked against the PDF** ([arXiv 1904.02666](https://pdf.arxiv.org/pdf/1904.02666)). In wearable HAR, the same CNN scored 99.85% under 10-fold CV and 85.1% under LOSO ([Gholamiangonabadi et al., IEEE Access 2020](https://openaccess.library.uitm.edu.my/Record/doaj-9bea9b3dbf2c4991ab92d75c8c999e3c)). The practice remains widespread ([Tello et al., arXiv 2310.11950](https://arxiv.org/abs/2310.11950)). The group key must be the physical trial ID, inherited by every window and every augmented copy.

Everything learned from data must be fitted inside the training fold: scalers, PCA, feature selection, class weights, augmentation and the fall-override threshold. Hyperparameters and window length are chosen by an inner grouped CV over the training subjects. Otherwise model selection leaks test information ([Kapoor & Narayanan](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)).

Little et al.'s commentary on Saeb is worth heeding as a nuance: the CV scheme must match the deployment question ([Little et al., GigaScience 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441396)). SafeSense-5G is meant for a new person in a known room, so LOSO is the matching protocol. LOSO with k-shot calibration matches a "calibrate at install" deployment. Neither says anything about a new room. A leave-one-furniture-configuration-out test on the empty-room trials is the only environment probe the data currently support. FallDeFi's fall from 93% to about 80% in changed rooms shows how much environment shift can cost ([FallDeFi](https://api.openalex.org/works/https://doi.org/10.1145/3161183)).

The empty-room design raises one more possible confound. If a mattress or chair is present during some activity trials but not others, the static channel could encode the class. Verifying that furniture configurations are balanced across classes, and applying static-baseline normalization, addresses this. This is an inference from the protocol, not a documented finding.

Metrics should be reported at three levels: window, trial (the unit of inference) and event. They should cover per-class precision, recall and F1, macro-F1, fall sensitivity, and false alarms per non-fall trial and, once continuous recordings exist, per hour.

Uncertainty must be reported honestly. Varoquaux shows that CV at around 100 samples carries error bars of about ±10%, and that the standard error across folds badly underestimates them ([Varoquaux, NeuroImage 180:68-77, 2018](https://arxiv.org/abs/1706.07581)). The report should include:

- per-subject scores and mean ± SD;
- cluster-bootstrap 95% CIs that resample trials within subjects (1,000-10,000 draws);
- Wilson intervals for fall recall and false-alarm rate;
- seed variance;
- a "model info sheet" documenting split procedure, class counts per fold (including subject 3's zero falls), empty-room handling, preprocessing order and any excluded data ([Kapoor & Narayanan](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)).

## Prioritized next steps

| # | Action | Done when |
|---|---|---|
| 1 | Drop the subject-3 duplication plan. Mark subject 3's fold "falls N/A, false-alarm rate only". Schedule ~20 real mattress falls for subject 3. | No relabelled data anywhere in the pipeline; recording session booked |
| 2 | Build the master trial index and data audit: per-RX native packet rate, interpolation fraction, shared exclusion mask, furniture configuration per trial. | One table drives every experiment |
| 3 | Fix labels: action windows only within 2-6 s (walk 2-7.5 s, hand 2-7 s); static windows labelled "still" or excluded; event timestamps for falls. | Fall label no longer covers stillness |
| 4 | Rebuild the evaluation harness: LOSO headline plus trial-grouped secondary; nested inner grouped CV; all fitting in-fold; trial-level aggregation; τ for the fall override tuned in inner folds; Wilson and cluster-bootstrap CIs. | Harness reproduces identical folds for every condition |
| 5 | Run the window and band ablation under LOSO: windows {0.5, 1, 2, 3} s; low-pass {10, 20, 40 Hz, none}; add 10-20 and 20-40 Hz band fractions, upper spectral edge and event-shape features. | Chosen window and cutoff justified by data |
| 6 | Run the model bake-off on the chosen features: logistic floor, HistGB, RF, SVM-RBF, small MLP; optional ROCKET-Ridge. Use class weights, not duplication. | Per-subject table with CIs |
| 7 | Run the 1-vs-2 RX matrix: P0 (E1-E4), then P1 controls (E5, C1-C3); two pre-registered hypotheses with McNemar, cluster bootstrap and Holm. | Fusion gain or null result with intervals |
| 8 | Run subject-invariance diagnostics: subject-ID probe, static-baseline normalization, scale-free feature variants. | Identity leakage quantified |
| 9 | Build the two-stage fall detector: trigger, event classifier, 2-4 s stillness and no-recovery confirmation; report the full sensitivity vs false-alarm curve. | Operating point chosen in inner folds |
| 10 | Run k-shot calibration curves (k = 1, 3, 5 trials per class from the held-out subject). | Zero-shot vs calibrated LOSO plotted |
| 11 | Collect new data: hard negatives (fast sit, lie-down, pick-up), continuous unscripted ADL for false alarms per hour, and more subjects of varied body size. | False-alarm claim covers realistic negatives |
| 12 | Optional: deep arm (E6/E7) and robustness probes (R1-R3). | Secondary results only |

## Conclusion

The literature reframes what SafeSense-5G is actually testing. Model families are close to interchangeable at this scale. What distinguishes credible CSI fall work from the inflated majority is the unit of analysis: events and trials, not 0.5 s windows. The other distinguishing factors are a frequency band matched to fall kinematics, and an evaluation that never lets a test person, trial or duplicate leak into training. The strongest fall systems won by changing the representation to speed and deceleration, not the classifier. Single-antenna ESP32 hardware rules out the phase-coherent version of that idea. An amplitude-based, event-segmented approximation is still within reach.

Two findings would be genuinely publishable whatever their direction. One is a capacity-controlled, LOSO-evaluated 1-vs-2 ESP32 receiver comparison for falls, which the notes found nowhere in the 2018-2026 literature. The other is an honest LOSO result with intervals, even if it lands short of 95/3, because it would be among the first rigorous ESP32 fall baselines. Before claiming deployment-grade specificity, the team should add fall-like negatives and continuous ADL recordings. The current class set omits exactly the activities that fool fall detectors.
