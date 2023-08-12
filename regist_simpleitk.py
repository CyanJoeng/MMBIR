from pathlib import Path
from sys import argv
import numpy as np

import cv2

from dnn.model_parameters import *
from utils.dataset import load_image

import SimpleITK as sitk


def data_pipeline(dataset_folder, data_id):
    data_dir = Path(dataset_folder) / "H&E_IMC" / "Pair" / data_id
    data_dir = Path(data_dir).absolute()
    main_id = str(data_id).split("_")[0]

    img_path = str(data_dir / f"HE{main_id}.tif")
    img_he = load_image(img_path, verbose=True, downsize=DOWN_SIZE)

    img_path = str(data_dir / f"{main_id}_panorama.tif")
    img_pano = load_image(img_path, verbose=True, downsize=DOWN_SIZE)

    print(f"Data id {data_id} image size he:{img_he.shape} pano:{img_pano.shape}")

    cv2.imwrite("/tmp/moving.tif", img_pano)
    cv2.imwrite("/tmp/fixed.tif", img_he)

    return img_pano, img_he


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


def command_multi_iteration(method):
    print("--------- Resolution Changing ---------")


def command_multiresolution_iteration(method):
    print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
    print("============= Resolution Change =============")


def fun_7(fixed, moving):
    R = sitk.ImageRegistrationMethod()

    R.SetShrinkFactorsPerLevel([3, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 1])

    R.SetMetricAsCorrelation()
    # R.SetMetricAsJointHistogramMutualInformation(20)
    R.MetricUseFixedImageGradientFilterOff()

    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        estimateLearningRate=R.EachIteration,
    )
    R.SetOptimizerScalesFromPhysicalShift()

    # initialTx = sitk.CenteredTransformInitializer(
    #     fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    # )
    # R.SetInitialTransform(initialTx)

    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform())
    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    R.AddCommand(
        sitk.sitkMultiResolutionIterationEvent,
        lambda: command_multiresolution_iteration(R),
    )

    outTx = R.Execute(fixed, moving)
    return outTx, R


def fun_6(fixed, moving):
    transformDomainMeshSize = [10] * moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

    print("Initial Parameters:")
    print(tx.GetParameters())

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetOptimizerAsGradientDescentLineSearch(
        5.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetShrinkFactorsPerLevel([6, 2, 1])
    R.SetSmoothingSigmasPerLevel([6, 2, 1])

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    R.AddCommand(
        sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R)
    )

    outTx = R.Execute(fixed, moving)
    return outTx, R


def fun_5(fixed, moving):
    transformDomainMeshSize = [8] * moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

    print("Initial Parameters:")
    print(tx.GetParameters())

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    return outTx, R


def fun_4(fixed, moving, args=[]):
    numberOfBins = 24
    samplingPercentage = 0.10

    if len(args) > 1:
        numberOfBins = int(args[2])
    if len(args) > 2:
        samplingPercentage = float(args[3])

    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsMattesMutualInformation(numberOfBins)

    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)

    R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 200)

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    return outTx, R


def fun_3(fixed, moving):
    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsCorrelation()
    # R.SetMetricAsMattesMutualInformation(24)
    R.SetMetricSamplingPercentage(0.1, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()

    initialTx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    )
    R.SetInitialTransform(initialTx)

    # tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
    # R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    return outTx, R


def fun_2(fixed, moving):
    fixed = sitk.Normalize(fixed)
    fixed = sitk.DiscreteGaussian(fixed, 2.0)

    moving = sitk.Normalize(moving)
    moving = sitk.DiscreteGaussian(moving, 2.0)

    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsJointHistogramMutualInformation()

    R.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-5,
        convergenceWindowSize=5,
    )

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    return outTx, R


def fun_1(fixed, moving):
    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsMeanSquares()

    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    return outTx, R


def write_log(outTx, R):
    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")


def get_overley(fixed, moving, outTx, out_path):
    sitk.WriteTransform(outTx, out_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)

    return_images = {"fixed": fixed, "moving": moving, "composition": cimg}

    # Calculate registration result scores
    # mse = sitk.MeanSquaredError(fixed, out)
    # ncc = sitk.GetNormalizedCrossCorrelation(fixed, out)
    # Convert SimpleITK images to NumPy arrays
    fixed_array = sitk.GetArrayFromImage(fixed)
    transformed_array = sitk.GetArrayFromImage(out)

    # Calculate mean squared error
    mse = np.mean((fixed_array - transformed_array) ** 2)
    # Calculate normalized cross correlation
    ncc = np.corrcoef(fixed_array.flatten(), transformed_array.flatten())[0, 1]
    ncc = -(ncc**2)

    score = {"mse": mse, "ncc": ncc}
    print(f"score: {score}")

    return return_images, score


if __name__ == "__main__":
    if len(argv) != 4:
        print(f"Usage: {argv[0]} dataset_folder data_id fun_id")
        exit(-1)

    dataset_dir = argv[1]
    data_id = argv[2]
    fun_id = int(argv[3]) - 1

    moving_image, fixed_image = data_pipeline(dataset_dir, data_id)

    fixed = sitk.ReadImage("/tmp/fixed.tif", sitk.sitkFloat32)
    moving = sitk.ReadImage("/tmp/moving.tif", sitk.sitkFloat32)

    outTx, R = [fun_1, fun_2, fun_3, fun_4, fun_5, fun_6, fun_7][fun_id](fixed, moving)

    write_log(outTx, R)

    imgs, score = get_overley(fixed, moving, outTx, f"outputs/itk/out_{data_id}.txt")
    sitk.WriteImage(imgs["composition"], f"outputs/itk/out_{data_id}.tif")
