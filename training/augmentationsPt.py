''' This file reutrns a set of composition of augmentations '''
from torchvision.transforms import v2


def Train_aug1():

    '''
    This function returns a transform object that is a composition of a RandomChoice object, which returns one of 5 equiprobable transformation objects

    '''
    transform_Aug = v2.Compose([ v2.RandomChoice([v2.RandomHorizontalFlip(p = 1),
                                                  v2.RandomVerticalFlip(p = 1),
                                                  v2.RandomAffine(degrees = 0, translate = (0.041,0.041), fill = 0),
                                                  v2.RandomRotation(degrees = 10, fill = 0),
                                                  v2.GaussianNoise(mean = 0, sigma = 0.03)
                                                ])
                             ])
    return transform_Aug


def Train_aug2():
    '''
	This function returns a transform object that is a composition of a multiple RandomChoice objects. It successively applies 5 different transforms 
	randomly. Each of the transform have a 0.25 probability of being picked.
    '''
    transform_Aug = v2.Compose([ v2.RandomChoice([v2.RandomHorizontalFlip(p = 0), v2.RandomHorizontalFlip(p = 1)],[0.75,0.25]),
                                 v2.RandomChoice([v2.RandomHorizontalFlip(p = 0), v2.RandomVerticalFlip(p = 1)],[0.75,0.25]),
                                 v2.RandomChoice([v2.RandomHorizontalFlip(p = 0), v2.RandomAffine(degrees = 0, translate = (0.041,0.041), fill = 0)],[0.75,0.25]),
                                 v2.RandomChoice([v2.RandomHorizontalFlip(p = 0), v2.RandomRotation(degrees = 10, fill = 0)],[0.75,0.25]),
                                 v2.RandomChoice([v2.RandomHorizontalFlip(p = 0), v2.GaussianNoise(mean = 0, sigma = 0.03)],[0.75,0.25])
                               ])
    return transform_Aug
