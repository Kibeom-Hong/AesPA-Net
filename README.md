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
