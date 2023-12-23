# DeepL-LIDC

Advance Techniques in Data Analysis Project.

## Motivation

Lung cancer is one of the most common types of cancer and is responsible for nearly 2 million deaths worldwide. Also, even in its early stages, lung cancer tends to be aggressive and there is always the possibility of resurgence. Due to these alarming numbers, there is an urgent need for new detection methods of lung cancer that can help health professionals improve the healthcare given to patients with lung cancer. Artficial Intelligence (AI) is a field that has gained recogniton due to its many possibilities and applications in the health field. Clinicians use Computed Tomography (CT) scans to assess the nodules and the likelihood of malignancy. Automatic solutions can help to make a faster and more accurate diagnosis, which is crucial for the early detection of lung cancer. Convolutional neural networks (CNN) based approaches have been shown to provide a reliable feature extraction ability to detect the malignancy risk associated with pulmonary nodules to classify the lung nodules.

## Objective

The objectve of this project is to use a public dataset with CT images (LIDC database
at <https://wiki.cancerimagingarchive.net/pages/viewpage.ac@on?pageId=1966254>) and develop a
learning model to classify lung nodules as malignant or benign. It is proposed to employ a convolutional neural network (CNN), or a deep neural network (DNN).

## Environment management

Create environment from `environment.yml` file:

```bash
conda env create --file environment.yml
```

It is recommended to edit the `environment.yml` mannualy when new packages are added, or unnecesary packages are removed to track updates with version control. To update the environment run:

```bash
conda env update --file environment.yml --prune
```

More info can be found here: <https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/04-sharing-environments/index.html>

## Install this package

```bash
pip install -e .
```

`.` indicates that we’re installing the package in the current directory. `-e` indicates that the package should be editable.

## Dependencies

- Pylidc ⚠️ (is no more mantained)

Based on the activity of official `Pylidc` repository <https://github.com/notmatthancock/pylidc>, it looks like it is no more mantained. To fix it, we can use a more updated version where `numpy` is upgraded: <https://github.com/vankhoa21991/pylidc>. Clone this repository using following command:

- Pytorch

Taking into consideration that we will work only on our local machines, we will install *CPU-only* version

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

```bash
git clone https://github.com/vankhoa21991/pylidc
```

, and install this version using:

```bash
pip install .
```

when you are in corresponding directory.

## Contributing

Fork the project repository to your GitHub account. This creates your own copy of the project.

Clone the forked repository to your local machine using the command:

```bash
git clone https://github.com/your-username/repository-name.git
```

Navigate into repository.

Create branch

```bash
git checkout -b your-branch-name
```

Make changes and commit them.

Push the changes to your fork:

```bash
git push origin your-branch-name
```

## 📚 References

[1] Silva, F.; Pereira, T.; Frade, J.; Mendes, J.; Freitas, C.; Hespanhol, V.; Costa, J L.; Cunha, A.; Oliveira, H.P. Pre-Training Autoencoder for Lung Nodule Malignancy Assessment Using CT Images. Appl. Sci. 2020, 10, 7837. <https://doi.org/10.3390/app10217837>

[2] Shen, W.; Zhou, M.; Yang, F.; Yang, C.; Tian, J. Mul@-Scale Convolu@onal Neural Networks for Lung Nodule Classiﬁca@on; Lecture Notes in Computer Science (Including Subseries Lecture Notes in Ar@ﬁcial Intelligence and Lecture Notes in Bioinforma@cs); Springer: Cham, Switzerland, 2015.

[3] Yan, X.; Pang, J.; Qi, H.; Zhu, Y.; Bai, C.; Geng, X.; Liu, M.; Terzopoulos, D.; Ding, X. Classiﬁca@on of lung nodule malignancy risk on computed tomography images using convolu@onal neural network: A comparison between 2d and 3d strategies. In Proceedings of the Asian Conference on Computer Vision, Taipei, Taiwan, 20–24 November 2016; pp. 91–101.
