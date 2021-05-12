# Single-Image-Super-Resolution-usign-SRGAN
## converting low resolution images (64 x 64) to high resolution images (256 x 256) ##
This project is about converting low resolution images to high resolution images using **Super Resolution Generative Adversarian Networks (SRGANs)**.

we are converting 64x64 low resolution images to 256x256 high resolution images.
we have used adversarial and VGG loss combined to train the model.

here you can find Generator and Discriminator model trained to **30000** epochs on **CelebA**.

## CelebA ##
CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K high ressolution celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

* 10,177 number of identities,

* 202,599 number of face images, and

* 5 landmark locations, 40 binary attributes annotations per image.


## Deployment ##

Model is deployed using Flask framework which you can find inside Flask folder of the Repo.

## Pre Trained Models ##
In Models folder you can find Generator model trained upto **30000 epochs** and for Discriminator follow belowed link
* Discriminator : https://drive.google.com/file/d/1apVR3DH3PqieWWqpIHu7mg5dvu-6t4BR/view?usp=sharing
* Generator : https://drive.google.com/file/d/1BXoeRCByfSSTySRh5pICfLhBSlJ9Q1Fw/view?usp=sharing

you can trin it further to get better accuracy.
