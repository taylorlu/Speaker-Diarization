# Speaker-Diarization

This project contains:
* Text-independent Speaker Identification module based on VGG-Speaker-recognition
* Speaker diarization based on UIS-RNN.

## Prerequisites
1. pytorch
2. keras
3. Tensorflow

## Outline
### 1. Speaker recognition.</br>
    cd ghostvlad
    python predict.py

The confusion matrix of 4 persons utterances is as below

    0.00  0.32  0.40  | 0.70  0.62  0.76  | 0.81  0.83  0.76  | 0.92  0.83  0.89  |

    0.32  0.00  0.48  | 0.68  0.58  0.76  | 0.87  0.84  0.83  | 0.92  0.82  0.86  |

    0.40  0.48  0.00  | 0.71  0.65  0.74  | 0.79  0.81  0.72  | 0.90  0.84  0.85  |

    ********************************************************************************

    0.70  0.68  0.71  | 0.00  0.35  0.30  | 0.78  0.81  0.76  | 0.80  0.81  0.80  |

    0.62  0.58  0.65  | 0.35  0.00  0.45  | 0.76  0.71  0.73  | 0.82  0.77  0.77  |

    0.76  0.76  0.74  | 0.30  0.45  0.00  | 0.83  0.83  0.80  | 0.83  0.84  0.80  |

    ********************************************************************************

    0.81  0.87  0.79  | 0.78  0.76  0.83  | 0.00  0.40  0.46  | 0.76  0.80  0.86  |

    0.83  0.84  0.81  | 0.81  0.71  0.83  | 0.40  0.00  0.45  | 0.80  0.78  0.82  |

    0.76  0.83  0.72  | 0.76  0.73  0.80  | 0.46  0.45  0.00  | 0.85  0.85  0.84  |

    ********************************************************************************

    0.92  0.92  0.90  | 0.80  0.82  0.83  | 0.76  0.80  0.85  | 0.00  0.41  0.44  |

    0.83  0.82  0.84  | 0.81  0.77  0.84  | 0.80  0.78  0.85  | 0.41  0.00  0.41  |

    0.89  0.86  0.85  | 0.80  0.77  0.80  | 0.86  0.82  0.84  | 0.44  0.41  0.00  |

    ********************************************************************************

Thanks to the authors of VGG, they are kind enough to provide the code and pre-trained model.
Their paper can refer to [UTTERANCE-LEVEL AGGREGATION FOR SPEAKER RECOGNITION IN THE WILD](https://arxiv.org/pdf/1902.10107.pdf)</br>
It's a novel idea that combines `netvlad/ghostvlad` which popularly used in image recognition to speaker recognition, the state-of-the-art in the past was `i-vector` based, which depended on the `GMM` model and `pLDA`.

This project only shows how to generate speaker embeddings using pre-trained model for uis-rnn training in later.</br>
The training project link to [VGG-Speaker-Recognition](https://github.com/WeidiXie/VGG-Speaker-Recognition) or [VGG-Speaker-Recognition](https://github.com/taylorlu/VGG-Speaker-Recognition)</br>
### Dataset
  http://www.openslr.org/38 contains 855 speakers and 120 utterances of Chinese Mandarin in each, so there are 102600 utterances in total.
  Generated speaker embeddings matrix can be found in ..
 
