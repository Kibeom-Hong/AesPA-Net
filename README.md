# AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks

### Official Pytorch Implementation of 'AesPA-Net' (ICCV 2023)
##### (Note that this project is totally powerd by Yonsei University)
![teaser](https://github.com/Kibeom-Hong/AesPA-Net/assets/77425614/8653065b-9554-4481-8673-caa797dab6e2)




> ## AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks
>
>  Paper[CVF] : TBD
>  
>  Paper[Arxiv] : [Link](https://arxiv.org/abs/2307.09724)
> 
> **Abstract**: To deliver the artistic expression of the target style, recent studies exploit the attention mechanism owing to its ability to map the local patches of the style image to the corresponding patches of the content image. However, because of the low semantic correspondence between arbitrary content and artworks, the attention module repeatedly abuses specific local patches from the style image, resulting in disharmonious and evident repetitive artifacts. To overcome this limitation and accomplish impeccable artistic style transfer, we focus on enhancing the attention mechanism and capturing the rhythm of patterns that organize the style. In this paper, we introduce a novel metric, namely pattern repeatability, that quantifies the repetition of patterns in the style image. Based on the pattern repeatability, we propose Aesthetic Pattern-Aware style transfer Networks (AesPA-Net) that discover the sweet spot of local and global style expressions. In addition, we propose a novel self-supervisory task to encourage the attention mechanism to learn precise and meaningful semantic correspondence. Lastly, we introduce the patch-wise style loss to transfer the elaborate rhythm of local patterns. Through qualitative and quantitative evaluations, we verify the reliability of the proposed pattern repeatability that aligns with human perception, and demonstrate the superiority of the proposed framework.

## Prerequisites

### Dependency
- Python==3.7
- CUDA==11.1
- Pytorch==1.7.1
- numpy==1.19.2
- Pillow==8.0.1
- imageio==2.9.0
- scipy==1.5.2


## Usage
#### Set pretrained weights
* Pretrained models for encoder(VGG-19) can be found in the `./baseline_checkpoints`
  -  Domainnes Indicator can be downloaded at [vgg_normalised_conv5_1.t7](https://drive.google.com/drive/folders/1HsJNskEMC5HUimq6ixkSZk7W_hgFNp7J?usp=sharing)
- Prepare pretrained models for **AesPA-Net**
  -  Domainnes Indicator can be downloaded at [Decoder.pth and Transformer.pth](https://drive.google.com/file/d/1-rf2CdrCr9ei9KS-V0H3kjo1oaPmT5Xz/view?usp=sharing)

- Move these pretrained weights to each folders:
  - transformer.pth -> `./train_results/<comment>/log/transformer_model.pth`
  - decoder.pth -> `./train_results/<comment>/log/dec_model.pth`

#### Inference (Automatic)
```
bash scripts/test_styleaware_v2.sh
```
or
```
python main.py --type test --batch_size #batch_size --comment <comment> --content_dir <content_dir> --style_dir <style_dir> --num_workers #num_workers
```

#### Training
```
bash scripts/train_styleaware_v2.sh
```


#### Evaluation
Available soon


## Citation
If you find this work useful for your research, please cite:
```
@article{Hong2023AesPANetAP,
  title={AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks},
  author={Kibeom Hong and Seogkyu Jeon and Junsoo Lee and Namhyuk Ahn and Kunhee Kim and Pilhyeon Lee and Daesik Kim and Youngjung Uh and Hyeran Byun},
  journal={ArXiv},
  year={2023},
  volume={abs/2307.09724},
  url={https://api.semanticscholar.org/CorpusID:259982728}
}
```

## Contact
If you have any question or comment, please contact the first author of this paper - Kibeom Hong

[kibeom9212@gmail.com](kibeom9212@gmail.com)