# UltraClean: Detect and Cleanse Poisoned Data to Train Backdoor-Free Neural Networks

Backdoor attacks are emerging threats to deep neural networks, which typically embed malicious behaviors into a victim model by injecting poisoned samples. The embedded backdoor can then be activated during inference by presenting the trigger on input images. Existing defenses have shown remarkable effectiveness in identifying and alleviating backdoor attacks since poisoned samples usually have suspicious patches, and the corresponding labels are often mislabeled. However, these defenses do not work for a recent new type of backdoor -- clean-label backdoor attacks that have imperceptible modification on poisoned samples and hold consistent labels. Defense against such stealthy attacks is less studied. In this paper, we propose UltraClean, a general framework that defends against both dirty-label and clean-label backdoor attacks. By measuring the susceptibility of training samples, UltraClean detects and removes poisons in the training dataset for eventually mitigating the backdoor effect. It achieves a high detection rate, significantly reduces the backdoor attack success rate while maintaining a decent model accuracy on clean data, outperforms existing defensive methods by a large margin. Moreover, UltraClean requires no prior knowledge about the process of data poisoning (e.g., which class is poisoned). Regardless of the dataset complexity, it can handle backdoor attacks with various poisoned data generation mechanisms.

![image](https://user-images.githubusercontent.com/36553004/157374971-4909986a-69f1-46d7-954a-4ced658757f7.png)


# Requirements

# Repo structure

The folder hierarchy of the <code>:

<code>
+-- dirty-label attacks
¦   +-- <models>
¦   +-- <Denoise>.py
¦   +-- <train>.py
¦   +-- <retrain.py>
¦   +-- <badNets_generation>.py
¦   +-- <blended_generation.py>
¦   +-- ...
+-- clean-label attacks
¦   +-- SIG
¦   ¦   +-- <poisoned_class>
¦   ¦   ¦   +-- <run>.py
¦   ¦   ¦   +-- <poison_generation>.py
¦   ¦   ¦   +-- <train>.py
¦   ¦   ¦   +-- <Denoise>.py
¦   ¦   ¦   +-- <SVD>.py
¦   ¦   ¦   +-- <test_Denoise>.py
¦   ¦   ¦   +-- <test_SVD>.py
¦   ¦   ¦   +-- <retrain>.py
¦   ¦   ¦   +-- ...
¦   ¦   +-- <whole_dataset>
¦   ¦   ¦   +-- <run_all>.py
¦   ¦   ¦   +-- <poison_generation>.py
¦   ¦   ¦   +-- <train>.py
¦   ¦   ¦   +-- <Denoise_allclasses>.py
¦   ¦   ¦   +-- <test_Denoise_allclasses>.py
¦   ¦   ¦   +-- <retrain>.py
¦   ¦   ¦   +-- ...
¦   +-- LCBD
¦   ¦   +-- <poisoned_class>
¦   ¦   ¦   +-- <run>.py
¦   ¦   ¦   +-- <generate_poisoned_dataset.py>.py
¦   ¦   ¦   +-- <train>.py
¦   ¦   ¦   +-- ...
¦   ¦   +-- <whole_dataset>
¦   ¦   ¦   +-- <run_all>.py
¦   ¦   ¦   +-- ...
¦   +-- HTBD
¦   ¦   +-- <poisoned_class>
¦   ¦   ¦   +-- <run>.py
¦   ¦   ¦   +-- <generate_poison.py>.py
¦   ¦   ¦   +-- <finetune_and_test>.py
¦   ¦   ¦   +-- ...
¦   ¦   +-- <whole_dataset>
¦   ¦   ¦   +-- <run_all>.py
¦   ¦   ¦   +-- ...

======================================================================================================
folder <dirty-label attacks>: contains all code of dirty-label attacks.

folder <models>: scripts of implementation of DNN models.

badNets_generation.py: script to generate poisons using badNets.

blended_generation.py: script to generate poisons using blended injection attack.

train.py: the script to train DNN models.

Denoise.py: the script to run the UltraClean framework for dirty-label attacks.

retrain.py: the script to retrain model on the sanitized dataset.

other python files: scripts of helper functions and utility functions for model training.

Trojan poisons generation follows: https://github.com/PurduePAML/TrojanNN

======================================================================================================
folder <clean-label attacks>: contains all code of clean-label attacks.

folder <poisoned_class>: contains all code for the experiment of detecion on the poisoned class.

folder <whole_dataset>:  contains all code for the experiment of detecion on the whole training dataset.

run.py: the all-in-one script to run the experiment flow of detection on the poisoned class.

run_all.py: the all-in-one script to run the experiment flow of detection on the whole training dataset.

poison_generation.py (SIG)/generate_poisoned_dataset.py (LCBD)/generate_poison.py (HTBD): scripts to generate poisoned samples (name varies because we keep the naming from the orignal repository).

train.py (SIG and LCBD)/finetune_and_test (HTBD): scripts to train DNN models.

Denoise.py: the script to run the UltraClean framework for detection on the poisoned class.

Denoise_allclasses.py: the script to run the UltraClean framework for detection on the whole training dataset.

SVD.py: the script to run SVD detection.

test_Denoise.py: the script to measure post-clean accuracy and ASR of UltraClean for detection on the poisoned class.

test_Denoise_allclasses.py: the script to measure post-clean accuracy and ASR of UltraClean for detection on the whole training dataset.

test_SVD.py: the script to measure post-clean accuracy and ASR of SVD for detection on the poisoned class.

retrain.py: the script to retrain model on the sanitized dataset.

other python files: scripts of helper functions and utility functions for SIG, LCBD and HTBD attacks.



# How to Run the Code

# Useful links
* BadeNets: https://github.com/Kooscii/BadNets
* Trojan: https://github.com/PurduePAML/TrojanNN
* Blended: https://arxiv.org/pdf/1712.05526.pdf
* SIG: https://github.com/DreamtaleCore/Refool
* LCBD: https://github.com/MadryLab/label-consistent-backdoor-code
* HTBD: https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
