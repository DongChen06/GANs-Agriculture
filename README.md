# GANs for Image Augmentation in Agriculture
In agricultural image analysis, optimal model performance is required for better fulfilling visual recognition tasks (e.g., classification, detection, and segmentation). Sufficient ground-truth datasets, however, are often difficult to obtain, posing a bottleneck to the development of advanced high-performance models. As artificial intelligence through deep learning is impacting analysis and modeling of agriculture images, data augmentation plays a vital role in boosting model performance by artificially generating new images to expand training datasets. Beyond traditional data augmentation techniques that rely on geometric and photometric transformations, generative adversarial network (GAN) invented in 2014 in the computer vision community, provides a suite of novel approaches that can learn good data representation and generate synthetic samples with the appearance of real images. Since 2018, there has been a growth of research into GANs for data augmentation or image synthesis for various agricultural applications. This repo contains GANs for augmenting agricultural images for improved model performance.


## Citation
Please consider cite our paper if you find this repo is helpful.


## Contents

* [Applications of GANs in Agriculture](#applications-of-gans-in-agriculture)
  * [Precision Agriculture](#precision-agriculture)
    * [Plant Health](#plant-health)
    * [Weeds](#weeds)
    * [Fruit Detection](#fruit-detection)
    * [Aquaculture](#aquaculture)
    * [Animal Farming](#animal-farming)
  * [Plant Phenotyping](#plant-phenotyping)
  * [Postharvest Quality Assessment](#postharvest-quality-assessment)
* [GAN Achitectures](#gan-achitectures)
* [GAN Review Papers](#gan-review-papers)

# Applications of GANs in Agriculture
## Precision Agriculture
### Plant Health

**2022**
 - Xu, M., Yoon, S., Fuentes, A., Yang, J., & Park, D. S. "Style-Consistent Image Translation: A Novel Data Augmentation Paradigm to Improve Plant Disease Recognition." Frontiers in Plant Science, 12(February), 1–16. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Style-Consistent+Image+Translation%3A+A+Novel+Data+Augmentation+Paradigm+to+Improve+Plant+Disease+Recognition&btnG=) [[paper]](https://www.frontiersin.org/articles/10.3389/fpls.2021.773142/full)

**2021**
- Abbas, Amreen, et al. "Tomato plant disease detection using transfer learning with C-GAN synthetic images." Computers and Electronics in Agriculture 187 (2021): 106279. [[scholar]](https://scholar.google.com/scholar?q=tomato+plant+disease+detection+using+transfer+learning+with+c-gan+synthetic+images&hl=en&as_sdt=0&as_vis=1&oi=scholart)  [[paper]](https://www.sciencedirect.com/science/article/pii/S0168169921002969)
- Hu, W. J., Xie, T. Y., Li, B. S., Du, Y. X., & Xiong, N. N. "An edge intelligence-based generative data augmentation system for IoT image recognition tasks." Journal of Internet Technology, 22(4), 765–777. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=An+edge+intelligence-based+generative+data+augmentation+system+for+IoT+image+recognition+tasks&btnG=) [[paper]](https://jit.ndhu.edu.tw/article/view/2541)
- Kim, C., Lee, H., & Jung, H., "Fruit tree disease classification system using generative adversarial networks." International Journal of Electrical and Computer Engineering, 11(3), 2508–2515. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Fruit+tree+disease+classification+system+using+generative+adversarial+networks&btnG=) [[paper]](http://ijece.iaescore.com/index.php/IJECE/article/view/23915)
- Zeng, M., Gao, H., & Wan, L. "Few-Shot Grape Leaf Diseases Classification Based on Generative Adversarial Network." Journal of Physics: Conference Series, 1883(1). [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Few-Shot+Grape+Leaf+Diseases+Classification+Based+on+Generative+Adversarial+Network.&btnG=) [[paper]](https://iopscience.iop.org/article/10.1088/1742-6596/1883/1/012093/meta)
- Gomaa, A. A., & El-Latif, Y. M. A. "Early Prediction of Plant Diseases using CNN and GANs." International Journal of Advanced Computer Science and Applications, 12(5), 514–519. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Early+Prediction+of+Plant+Diseases+using+CNN+and+GANs&btnG=) [[paper]](https://thesai.org/Publications/ViewPaper?Volume=12&Issue=5&Code=IJACSA&SerialNo=63) 
- Maqsood, M. H., Mumtaz, R., Haq, I. U., Shafi, U., Zaidi, S. M. H., & Hafeez, M. "Super resolution generative adversarial network (SRGANs) for wheat stripe rust classification." Sensors, 21(23), 1–12. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Super+resolution+generative+adversarial+network+%28SRGANs%29+for+wheat+stripe+rust+classification&btnG=) [[paper]](https://www.mdpi.com/1424-8220/21/23/7903)
- Zhang, Jingyao, Rao, Y., Man, C., Jiang, Z., & Li, S. "Identification of cucumber leaf diseases using deep learning and small sample size for agricultural Internet of Things." International Journal of Distributed Sensor Networks, 17(4). [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Identification+of+cucumber+leaf+diseases+using+deep+learning+and+small+sample+size+for+agricultural+Internet+of+Things.&btnG=) [[paper]](https://journals.sagepub.com/doi/full/10.1177/15501477211007407)
- Zhao, Y., Chen, Z., Gao, X., Song, W., Xiong, Q., Hu, J., & Zhang, Z. "Plant Disease Detection using Generated Leaves Based on DoubleGAN." IEEE/ACM Transactions on Computational Biology and Bioinformatics, 5963(c), 1–10. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Plant+Disease+Detection+using+Generated+Leaves+Based+on+DoubleGAN&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9345997?casa_token=lmemnHLWZ0MAAAAA:ypbV6WIy5pOnt15iZ5TJDW4ip2zsGTXBXY2wp24vjwk0rmLwd3QCWzDlWETnrnC6-9s9RAbenA)
- Deng, H., Luo, D., Chang, Z., Li, H., & Yang, X. "Rahc_gan: A data augmentation method for tomato leaf disease recognition." Symmetry, 13(9). [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Rahc_gan%3A+A+data+augmentation+method+for+tomato+leaf+disease+recognition.&btnG=) [[paper]](https://www.mdpi.com/2073-8994/13/9/1597)
- Nerkar, B., & Talbar, S. "Cross-dataset learning for performance improvement of leaf disease detection using reinforced generative adversarial networks." International Journal of Information Technology (Singapore), 13(6), 2305–2312. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Cross-dataset+learning+for+performance+improvement+of+leaf+disease+detection+using+reinforced+generative+adversarial+networks&btnG=) [[paper]](https://link.springer.com/article/10.1007/s41870-021-00772-1)

**2020**
- Wu, Q., Chen, Y., & Meng, J. "DCGAN-based data augmentation for tomato leaf disease identification." IEEE Access, 8, 98716–98728.[[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=DCGAN-based+data+augmentation+for+tomato+leaf+disease+identification&btnG=)  [[paper]](https://ieeexplore.ieee.org/abstract/document/9099295)
- Sun, R., Zhang, M., Yang, K., & Liu, J., "Data enhancement for plant disease classification using generated lesions." Applied Sciences (Switzerland), 10(2).[[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Data+enhancement+for+plant+disease+classification+using+generated+lesions&btnG=) [[paper]](https://www.mdpi.com/2076-3417/10/2/466)
- Zeng, Q., Ma, X., Cheng, B., Zhou, E., & Pang, W. "GANS-based data augmentation for citrus disease severity detection using deep learning." IEEE Access, 8, 172882–172891. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=GANS-based+data+augmentation+for+citrus+disease+severity+detection+using+deep+learning&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9200543)
- Yuwana, R. S., Fauziah, F., Heryana, A., Krisnandi, D., Kusumo, R. B. S., & Pardede, H. F., "Data Augmentation using Adversarial Networks for Tea Diseases Detection." Jurnal Elektronika Dan Telekomunikasi, 20(1), 29. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Data+Augmentation+using+Adversarial+Networks+for+Tea+Diseases+Detection&btnG=) [[paper]](https://www.jurnalet.com/jet/article/view/365)
- Cap, Q. H., Uga, H., Kagiwada, S., & Iyatomi, H."LeafGAN: An Effective Data Augmentation Method for Practical Plant Disease Diagnosis." IEEE Transactions on Automation Science and Engineering, 1–10. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=LeafGAN%3A+An+Effective+Data+Augmentation+Method+for+Practical+Plant+Disease+Diagnosis&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9298454?casa_token=aRd68BhDTIMAAAAA:gk9OD1CC49QS5F0pFs7_7O2ZVoP3Ef6rkkUzrdX4Rp-cMD9KHdw1-7QsrENjccqONMVsKc2Xpg)
- Wen, J., Shi, Y., Zhou, X., & Xue, Y. "Crop disease classification on inadequate low-resolution target images." Sensors (Switzerland), 20(16), 1–17. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Crop+disease+classification+on+inadequate+low-resolution+target+images&btnG=) [[paper]](https://www.mdpi.com/1424-8220/20/16/4601)
- Nazki, H., Yoon, S., Fuentes, A., & Park, D. S. "Unsupervised image translation using adversarial networks for improved plant disease recognition." Computers and Electronics in Agriculture, 168(August 2019), 105117. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Unsupervised+image+translation+using+adversarial+networks+for+improved+plant+disease+recognition&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S0168169919315339?casa_token=_UNcIX59cG4AAAAA:A7X0yaHeFCCC1rFGWhIK5AjxDscVFhS0dXWdf3IN0qNrtdk7mREdu0GiXeL_P_oNcA5BpmZWA90)
- Liu, B., Tan, C., Li, S., He, J., & Wang, H. "A Data Augmentation Method Based on Generative Adversarial Networks for Grape Leaf Disease Identification." IEEE Access, 8, 102188–102198. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=A+Data+Augmentation+Method+Based+on+Generative+Adversarial+Networks+for+Grape+Leaf+Disease+Identification.&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9104723)
- Dai, Q., Cheng, X., Qiao, Y., & Zhang, Y. "Agricultural pest super-resolution and identification with attention enhanced residual and dense fusion generative and adversarial network." IEEE Access, 8, 81943–81959. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Agricultural+pest+super-resolution+and+identification+with+attention+enhanced+residual+and+dense+fusion+generative+and+adversarial+network.&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9082695) 
- Dai, Q., Cheng, X., Qiao, Y., & Zhang, Y.  "Crop leaf disease image super-resolution and identification with dual attention and topology fusion generative adversarial network." IEEE Access, 8, 55724–55735. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Crop+leaf+disease+image+super-resolution+and+identification+with+dual+attention+and+topology+fusion+generative+adversarial+network.&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9042295)
- Bi, L., & Hu, G. "Improving Image-Based Plant Disease Classification With Generative Adversarial Network Under Limited Training Set." Frontiers in Plant Science, 11(December), 1–12. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Improving+Image-Based+Plant+Disease+Classification+With+Generative+Adversarial+Network+Under+Limited+Training+Set&btnG=) [[paper]](https://www.frontiersin.org/articles/10.3389/fpls.2020.583438/full)

**2019**
- Hu, G., Wu, H., Zhang, Y., & Wan, M., 2019. A low shot learning method for tea leaf’s disease identification. Computers and Electronics in Agriculture, 163(June). [[scholar]](https://doi.org/10.1016/j.compag.2019.104852)
- Douarre, C., Crispim-Junior, C. F., Gelibert, A., Tougne, L., & Rousseau, D., 2019. Novel data augmentation strategies to boost supervised segmentation of plant disease. Computers and Electronics in Agriculture, 165(August), 104967.[[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Novel+data+augmentation+strategies+to+boost+supervised+segmentation+of+plant+disease&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S0168169919304879?casa_token=vbGZFiKsXlMAAAAA:Nf6D7d4Y9hOng_oSmFnKAnwhnzCrc7aKo9KjHkyP7ie58_IRb6QCx-tSFvedkWUinoNRncDPg4w)
- Zhang, M., Liu, S., Yang, F., & Liu, J. "Classification of Canker on Small Datasets Using Improved Deep Convolutional Generative Adversarial Networks." IEEE Access, 7, 49680–49690. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Classification+of+Canker+on+Small+Datasets+Using+Improved+Deep+Convolutional+Generative+Adversarial+Networks&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/8648422)
- Nazki, H., Lee, J., Yoon, S., & Park, D. S. "Image-to-Image Translation with GAN for Synthetic Data Augmentation in Plant Disease Datasets." Korean Institute of Smart Media, 8(2), 46–57. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Image-to-Image+Translation+with+GAN+for+Synthetic+Data+Augmentation+in+Plant+Disease+Datasets&btnG=) [[paper]](https://www.koreascience.or.kr/article/JAKO201918961949570.page)
- Tian, Y., Yang, G., Wang, Z., Li, E., & Liang, Z. "Detection of apple lesions in orchards based on deep learning methods of cyclegan and YoloV3-dense." Journal of Sensors, 2019. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Detection+of+apple+lesions+in+orchards+based+on+deep+learning+methods+of+cyclegan+and+YoloV3-dense&btnG=) [[paper]](https://www.hindawi.com/journals/js/2019/7630926/)
- Arsenovic, M., Karanovic, M., Sladojevic, S., Anderla, A., & Stefanovic, D. "Solving current limitations of deep learning based approaches for plant disease detection." Symmetry, 11(7). [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Solving+current+limitations+of+deep+learning+based+approaches+for+plant+disease+detection.&btnG=) [[paper]](https://www.mdpi.com/2073-8994/11/7/939)
- Lu, C. Y., Arcega Rustia, D. J., & Lin, T. Te. "Generative Adversarial Network Based Image Augmentation for Insect Pest Classification Enhancement." IFAC-PapersOnLine, 52(30), 1–5. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Generative+Adversarial+Network+Based+Image+Augmentation+for+Insect+Pest+Classification+Enhancement&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S2405896319323109) 


### Weed Controls

**2021**
 - Fawakherji, M., Potena, C., Pretto, A., Bloisi, D. D., & Nardi, D. "Multi-Spectral Image Synthesis for Crop/Weed Segmentation in Precision Farming." Robotics and Autonomous Systems, 146, 103861. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Multi-Spectral+Image+Synthesis+for+Crop%2FWeed+Segmentation+in+Precision+Farming&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S0921889021001469?casa_token=q36tcKscSUsAAAAA:i8tmtC4TLYG1xQ1O2jaPrEuX8DgY2iEP58yylNF5ww42TlfNetxcdNNtGtLqurKxJnSX_RXnxyM)
 -  Espejo-Garcia, B., Mylonas, N., Athanasakos, L., Vali, E., & Fountas, S. "Combining generative adversarial networks and agricultural transfer learning for weeds identification." Biosystems Engineering, 204, 79–89. [scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Combining+generative+adversarial+networks+and+agricultural+transfer+learning+for+weeds+identification.&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S1537511021000155?casa_token=zJTDYIS2_tgAAAAA:jjPxGVdPbbtJl3eC0x1r1yrcBRNixWSVTsgzeMhL6jXjzvsWzLHS2Q0l-Mwalya0W51BH-7wHoM)
 -  Khan, S, Tufail, M, Khan, M. T., Khan, Z. A., Iqbal, J., and Alam, M. “A novel semi-supervised framework for UAV based crop/weed classification.” PLoS One, vol. 16, no. 5 [[scholar]](https://scholar.google.com/scholar?q=A+novel+semi-supervised+framework+for+UAV+based+crop/weed+classification&hl=en&as_sdt=0,25) [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251008)

**2019**

 - Kerdegari,H., Razaak M., Argyriou V., Remagnino P. "Semi-supervised GAN for Classification of Multispectral Imagery Acquired by UAVs." [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Semi-supervised+GAN+for+Classification+of+Multispectral+Imagery+Acquired+by+UAVs&btnG=) [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251008)



### Fruit Detection
**2021**
- Fei, Z., Olenskyj, A., Bailey, B. N., & Earles, M. Enlisting 3D Crop Models and GANs for More Data Efficient and Generalizable Fruit Detection. 1269–1277. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=.+Enlisting+3D+Crop+Models+and+GANs+for+More+Data+Efficient+and+Generalizable+Fruit+Detection&btnG=) [[paper]](https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/html/Fei_Enlisting_3D_Crop_Models_and_GANs_for_More_Data_Efficient_ICCVW_2021_paper.html) 
- Kierdorf, J., Weber, I., Kicherer, A., Zabawa, L., Drees, L., & Roscher, R. "Behind the leaves -- Estimation of occluded grapevine berries with conditional generative adversarial networks." [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Behind+the+leaves+--+Estimation+of+occluded+grapevine+berries+with+conditional+generative+adversarial+networks.&btnG=) [[paper]](https://arxiv.org/abs/2105.10325) 

**2020**

 - Barth, R., Hemming, J., & Van Henten, E. J. "Optimising realism of synthetic images using cycle generative adversarial networks for improved part segmentation." Computers and Electronics in Agriculture, 173(October 2019), 105378. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Optimising+realism+of+synthetic+images+using+cycle+generative+adversarial+networks+for+improved+part+segmentation+&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S0168169919320794)
 - Luo Z., Huiling Y., Zhang Y. "Pine Cone Detection Using Boundary Equilibrium Generative Adversarial Networks and Improved YOLOv3 Model." Sensors 2020, 20(16), 4430. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Pine+Cone+Detection+Using+Boundary+Equilibrium+Generative+Adversarial+Networks+and+Improved+YOLOv3+Model&btnG=) [[paper]](https://www.mdpi.com/1424-8220/20/16/4430)
 - Bellocchio, E.,Costante G., Cascianelli, S., Fravolini,  M. L., Valigi P."Combining Domain Adaptation and Spatial Consistency for Unseen Fruits Counting: A Quasi-Unsupervised Approach." IEEE Robotics and Automation Letters,Volume: 5, Issue: 2, pp. 1079 - 1086. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Combining+Domain+Adaptation+and+Spatial+Consistency+for+Unseen+Fruits+Counting%3A+A+Quasi-Unsupervised+Approach&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/8957569)
 - Olatunji, J. R., Redding, G. P., Rowe, C. L., & East, A. R., 2020. Reconstruction of kiwifruit fruit geometry using a CGAN trained on a synthetic dataset. Computers and Electronics in Agriculture, 177(August), 105699. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Reconstruction+of+kiwifruit+fruit+geometry+using+a+CGAN+trained+on+a+synthetic+dataset&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S0168169920310206)
 

**2018**

 - Barth, R., Hemming, J., & Van Henten, E. J. 2018. Improved Part Segmentation Performance by Optimising Realism of Synthetic Images using Cycle Generative Adversarial Networks. arXiv:1803.06301v1  [cs.CV]  16 Mar 2018. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Improved+Part+Segmentation+Performance+by+Optimising+Realism+of+Synthetic+Images+using+Cycle+Generative+Adversarial+Networks.&btnG=) [[paper]](https://arxiv.org/abs/1803.06301)
 


### Aquaculture
**2021**

- Zhang, Junjie, Yang, G., Sun, L., Zhou, C., Zhou, X., Li, Q., Bi, M., & Guo, J., 2021. Shrimp egg counting with fully convolutional regression network and generative adversarial network. Aquacultural Engineering, 94(June), 102175. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Shrimp+egg+counting+with+fully+convolutional+regression+network+and+generative+adversarial+network&btnG=) [[paper]](https://www.sciencedirect.com/science/article/pii/S0144860921000315) 

**2018**
- Zhao, Jian, Li, Y., Zhang, F., Zhu, S., Liu, Y., Lu, H., & Ye, Z., 2018. Semi-Supervised Learning-Based Live Fish Identification in Aquaculture Using Modified Deep Convolutional Generative Adversarial Networks. Transactions of the ASABE, 61(2), 699–710. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=+Semi-Supervised+Learning-Based+Live+Fish+Identification+in+Aquaculture+Using+Modified+Deep+Convolutional+Generative+Adversarial+Networks&btnG=) [[paper]](https://elibrary.asabe.org/abstract.asp?aid=48905)

### Animal Farming
**2021**
- Ahmed, G., Malick, R. A. S., Akhunzada, A., Zahid, S.,  Sagri, M. R. and Gani, A. “An approach towards iot-based  predictive service for early detection of diseases in poultry chickens,” Sustain., vol. 13, no. 23, pp. 1–16, 2021. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=An+approach+towards+iot-based++predictive+service+for+early+detection+of+diseases+in+poultry+chickens&btnG=) [[paper]](https://www.mdpi.com/2071-1050/13/23/13396)
- Singh P., Devi K. J., Varish N. "Muzzle Pattern Based Cattle Identification Using Generative Adversarial Networks." Advances in Intelligent Systems and Computing Book series, vol. 1392. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Muzzle+Pattern+Based+Cattle+Identification+Using+Generative+Adversarial+Networks&btnG=) [[paper]](https://link.springer.com/chapter/10.1007/978-981-16-2709-5_2)

**2020**
- Li, H., & Tang, J. "Dairy goat image generation based on improved-self-attention generative adversarial." IEEE Access (Volume: 8) pp. 62448 - 62457. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C25&q=Dairy+goat+image+generation+based+on+improved-self-attention+generative+adversarial&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9039669)


## Plant Phenotyping

## Postharvest Quality Assessment


# GAN Achitectures

# GAN Review Papers
**2022**
- Cohen, Gilad, and Raja Giryes. "Generative Adversarial Networks." arXiv preprint arXiv:2203.00667 (2022). [[arXiv]](https://arxiv.org/abs/2203.00667) 
- 

**2021**
- Pavan Kumar, M. R., and Prabhu Jayagopal. "Generative adversarial networks: a survey on applications and challenges." International Journal of Multimedia Information Retrieval 10.1 (2021): 1-24. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&as_vis=1&q=Generative+adversarial+networks%3A+a+survey+on+applications+and+challenges.&btnG=) [[paper]](https://link.springer.com/article/10.1007/s13735-020-00196-w) 
- Gui, Jie, et al. "A review on generative adversarial networks: Algorithms, theory, and applications." IEEE Transactions on Knowledge and Data Engineering (2021). [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&as_vis=1&q=A+review+on+generative+adversarial+networks%3A+Algorithms%2C+theory%2C+and+applications&btnG=) [[paper]](https://ieeexplore.ieee.org/abstract/document/9625798?casa_token=pVTomSmhKXkAAAAA:6u9iUDfYyuu4If48_8P0Zv930tJFXoDBjDBlYfRuIkiCgOv71WMMOl0XnPTG-6rRXP-VnM33A2w) 
- Wang, Zhengwei, Qi She, and Tomas E. Ward. "Generative adversarial networks in computer vision: A survey and taxonomy." ACM Computing Surveys (CSUR) 54.2 (2021): 1-38. [[scholar]](https://scholar.google.com/scholar?q=A+review+on+generative+adversarial+networks:+Algorithms,+theory,+and+applications&hl=en&as_sdt=0&as_vis=1&oi=scholart) [[arXiv]](https://arxiv.org/abs/2001.06937) 

**2019**
- Di Mattia, Federico, et al. "A survey on gans for anomaly detection." arXiv preprint arXiv:1906.11632 (2019). [[scholar]](https://scholar.google.com/scholar?q=A+Survey+on+GANs+for+Anomaly+Detection&hl=en&as_sdt=0&as_vis=1&oi=scholart) [[arXiv]](https://arxiv.org/pdf/1906.11632.pdf)
- Yi, Xin, Ekta Walia, and Paul Babyn. "Generative adversarial network in medical imaging: A review." Medical image analysis 58 (2019): 101552. [[scholar]](https://scholar.google.com/scholar?q=Generative+adversarial+network+in+medical+imaging:+A+review&hl=en&as_sdt=0&as_vis=1&oi=scholart) [[arXiv]](https://www.sciencedirect.com/science/article/pii/S1361841518308430?casa_token=97Qge_RSfXoAAAAA:MqwKdeYIUPLqL_dkZubJCQ-cM3Jzj2jPX-flhejJjAWIWXuLxLNmIDEs7D1YI_uIqDgJlVTRyWM)
- Zamorski, Maciej, et al. "Generative adversarial networks: recent developments." International Conference on Artificial Intelligence and Soft Computing. Springer, Cham, 2019. [[scholar]](https://scholar.google.com/scholar?q=Generative+adversarial+networks:+recent+developments&hl=en&as_sdt=0&as_vis=1&oi=scholart) [[arXiv]](https://arxiv.org/pdf/1903.12266.pdf)
- Hong, Yongjun, et al. "How generative adversarial networks and their variants work: An overview." ACM Computing Surveys (CSUR) 52.1 (2019): 1-43. [[scholar]](https://scholar.google.com/scholar?q=How+generative+adversarial+networks+and+their+variants+work:+An+overview&hl=en&as_sdt=0&as_vis=1&oi=scholart) [[paper]](https://dl.acm.org/doi/abs/10.1145/3301282?casa_token=AnBrUIUDvtEAAAAA:P7MjRwo6fxppV7Jw1vQtzndZ_mkFbQbC65EO68PkZxAoqf9YF88uepg2y_7vkLxzI3qso8qwbuD1Eg)

**2018**
- Creswell, Antonia, et al. "Generative adversarial networks: An overview." IEEE Signal Processing Magazine 35.1 (2018): 53-65. [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&as_vis=1&q=Generative+adversarial+networks%3A+An+overview+A+Creswell&btnG=) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8253599&casa_token=xRIFFXWsCCMAAAAA:OtCGlWJGo7KEp6j2ubuvod3qBW9Raaw3Rp_qhl7MM6umT4gruBC2cj8ut3DiIjK06_55NWlGfSM&tag=1) [[arXiv]](https://arxiv.org/abs/1710.07035)
- Cao, Yang-Jie, et al. "Recent advances of generative adversarial networks in computer vision." IEEE Access 7 (2018): 14985-15006. [[scholar]](https://scholar.google.com/scholar?q=Recent+advances+of+generative+adversarial+networks+in+computer+vision&hl=en&as_sdt=0&as_vis=1&oi=scholart) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8576508)
