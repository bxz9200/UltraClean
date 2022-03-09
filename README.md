# UltraClean: Detect and Cleanse Poisoned Data to Train Backdoor-Free Neural Networks

Backdoor attacks are emerging threats to deep neural networks, which typically embed malicious behaviors into a victim model by injecting poisoned samples. The embedded backdoor can then be activated during inference by presenting the trigger on input images. Existing defenses have shown remarkable effectiveness in identifying and alleviating backdoor attacks since poisoned samples usually have suspicious patches, and the corresponding labels are often mislabeled. However, these defenses do not work for a recent new type of backdoor -- clean-label backdoor attacks that have imperceptible modification on poisoned samples and hold consistent labels. Defense against such stealthy attacks is less studied. In this paper, we propose UltraClean, a general framework that defends against both dirty-label and clean-label backdoor attacks. By measuring the susceptibility of training samples, UltraClean detects and removes poisons in the training dataset for eventually mitigating the backdoor effect. It achieves a high detection rate, significantly reduces the backdoor attack success rate while maintaining a decent model accuracy on clean data, outperforms existing defensive methods by a large margin. Moreover, UltraClean requires no prior knowledge about the process of data poisoning (e.g., which class is poisoned). Regardless of the dataset complexity, it can handle backdoor attacks with various poisoned data generation mechanisms.

![image](https://user-images.githubusercontent.com/36553004/157374971-4909986a-69f1-46d7-954a-4ced658757f7.png)


# Requirements

# Repo Structure

![image](https://user-images.githubusercontent.com/36553004/157376363-e7f06f36-543d-4ad0-91d7-25ed73278ed1.png)



# How to Run the Code

# Useful links
* BadeNets: https://github.com/Kooscii/BadNets
* Trojan: https://github.com/PurduePAML/TrojanNN
* Blended: https://arxiv.org/pdf/1712.05526.pdf
* SIG: https://github.com/DreamtaleCore/Refool
* LCBD: https://github.com/MadryLab/label-consistent-backdoor-code
* HTBD: https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
