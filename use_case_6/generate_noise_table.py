import numpy as np
import pickle

# Laser parameters
laserDiameter = 4.5e-3
wavelength = 850e-9
laserDivergence = 0.2e-3

# Camera parameters
Hpixels = 1936
Vpixels = 1216
pxWidth = 5.86e-6
fl = 16e-3

# More parameters
laserPower = 100e-6
frameRate = 50
pulseEnergy = laserPower / frameRate

# Parameters for Lidar equation. Sigma includes objet reflectivity and objective lens transmissivity
Ar = np.pi * (11.4e-3 / 2) ** 2
sigma = 10 / 100

# More parameters
quantumEfficiency = 6.25 / 100

# Estimate the angular field of view of each pixel of the camera
Hfov = 2 * np.arctan(pxWidth * Hpixels / 2 / fl)
Vfov = 2 * np.arctan(pxWidth * Vpixels / 2 / fl)
HfovPx = Hfov / Hpixels

# Define constants
h = 6.626e-34  # J*s
c = 299792458  # m/s


def add_camera_noise(
    input_irrad_photons,
    qe=0.1029,
    sensitivity=1.923,
    dark_noise=6.83,
    bitdepth=12,
    baseline=100,
    rs=np.random.RandomState(seed=42),
    noiseOn=True,
):
    if noiseOn:
        # Add shot noise
        photons = rs.poisson(input_irrad_photons, size=input_irrad_photons.shape)

        # Convert to electrons
        electrons = qe * photons

        # Add dark noise
        electrons_out = rs.normal(scale=dark_noise, size=electrons.shape) + electrons

        # Convert to ADU and add baseline
        max_adu = int(2**bitdepth - 1)
        adu = (electrons_out * sensitivity).astype(int)  # Convert to discrete numbers
        adu += baseline
        adu[adu > max_adu] = max_adu  # models pixel saturation
    else:
        adu = (input_irrad_photons * qe * sensitivity).astype(int)

        # Convert to ADU and add baseline
        max_adu = int(2**bitdepth - 1)
        adu[adu > max_adu] = max_adu  # models pixel saturation

    return adu


def get_error(point, cx_aux, cy_aux, refleftivity=sigma, camera="left"):
    x, y, z, i = point.copy()

    # Left camera
    if camera == "left":
        r = np.sqrt(((x - (-0.6)) ** 2) + (y**2) + (z**2))
    elif camera == "right":
        r = np.sqrt(((x - 0.6) ** 2) + (y**2) + (z**2))
    # Estimate pixel width at a given distance z
    laserDotDiameter = laserDiameter + 2 * r * np.tan(laserDivergence / 2)
    pixelWidthAtz = 2 * r * np.tan(HfovPx / 2)

    # Generate circle based on point
    # Generate the laser circumference
    x = np.linspace(-laserDotDiameter / 2, +laserDotDiameter / 2, 256)
    y = np.sqrt(laserDotDiameter**2 / 4 - x**2)

    # Generate laser dot
    x = x + 0
    y = y + 0

    x = np.hstack((x, x[::-1]))
    y = np.hstack((y, -y))

    # Identify which pixels are illuminated by the laser dots
    x1minPx = np.floor(np.min(x / pixelWidthAtz) + cx_aux)
    x1maxPx = np.ceil(np.max(x / pixelWidthAtz) + cx_aux)
    y1minPx = np.floor(np.min(y / pixelWidthAtz) + cy_aux)
    y1maxPx = np.ceil(np.max(y / pixelWidthAtz) + cy_aux)

    # Identify the meshgrid of pixels
    xPx = np.arange(x1minPx, x1maxPx + 1)
    yPx = np.arange(y1minPx, y1maxPx + 1)
    XPx, YPx = np.meshgrid(xPx, yPx)

    # Fill the dot with dots
    Npoints = int(1e5)
    circleRadius = laserDotDiameter / pixelWidthAtz / 2
    circleDots = (
        np.random.rand(Npoints)
        * circleRadius
        * np.exp(1j * 2 * np.pi * np.random.rand(Npoints))
    )
    circleDotsx = np.real(circleDots) + np.mean(x) / pixelWidthAtz
    circleDotsy = np.imag(circleDots)

    # Generate laser dot
    circleDotsx += cx_aux
    circleDotsy += cy_aux

    # Shift x and y
    x += cx_aux * pixelWidthAtz
    y += cy_aux * pixelWidthAtz

    # Calculate number of photons in each pixel
    # Sweep each pixel and estimate the number of photons that fall on it
    dotsInPixel = np.zeros((len(yPx), len(xPx), Npoints), dtype=bool)

    # Loop
    for a in range(len(yPx)):
        for b in range(len(xPx)):
            pixelLimits = [
                XPx[a, b] - 0.5,
                XPx[a, b] + 0.5,
                YPx[a, b] - 0.5,
                YPx[a, b] + 0.5,
            ]

            # Identify the dots that fall on the pixel
            dotsInPixel[a, b, :] = np.logical_and(
                np.logical_and(
                    circleDotsx >= pixelLimits[0], circleDotsx <= pixelLimits[1]
                ),
                np.logical_and(
                    circleDotsy >= pixelLimits[2], circleDotsy <= pixelLimits[3]
                ),
            )

    # Count (and normalize)the number of dots in each pixel
    dotsInPixelCount = np.sum(dotsInPixel, axis=2) / Npoints

    # Pixelate
    # Total number of received photonics
    freeSpacePathLoss = 1 / (4 * np.pi * r**2)
    ERx = pulseEnergy * freeSpacePathLoss * Ar * refleftivity
    Ephoton = h * c / wavelength
    Nphotons = ERx / Ephoton

    # Total number of photons in each pixel
    NphotonsForEachPixel = Nphotons * dotsInPixelCount

    output_matrix = add_camera_noise(
        NphotonsForEachPixel, qe=quantumEfficiency, noiseOn=True
    )

    # Calculate centroid
    centroid_x = np.sum(XPx * output_matrix) / np.sum(output_matrix)
    centroid_y = np.sum(YPx * output_matrix) / np.sum(output_matrix)

    return [centroid_x, centroid_y]


# Store noise value in a dictionary
if __name__ == "__main__":
    # Create dictionary with keys for both cameras
    noise_values = {"left": {}, "right": {}}

    # Iterate over all distances (integer) between 2m and 70m
    points = [[0, 0, i, 1] for i in range(1, 71)]
    for point in points:
        print(point)

        # Iterate over every one decimal case combination of zero for X
        for cx_aux in np.arange(-0.5, 0.6, 0.1):
            cx_aux = np.round(cx_aux, 1)
            if cx_aux not in noise_values["left"]:
                noise_values["left"][cx_aux] = {}
                noise_values["right"][cx_aux] = {}

            # Iterate over every one decimal case combination of zero for Y
            for cy_aux in np.arange(-0.5, 0.6, 0.1):
                cy_aux = np.round(cy_aux, 1)
                if cy_aux not in noise_values["left"][cx_aux]:
                    noise_values["left"][cx_aux][cy_aux] = {}
                    noise_values["right"][cx_aux][cy_aux] = {}

                # Iterate ten times to store ten different values for each camera
                for i in range(10):
                    if point[2] in noise_values["left"][cx_aux][cy_aux]:
                        noise_values["left"][cx_aux][cy_aux][point[2]].append(
                            get_error(point, cx_aux, cy_aux, camera="left")
                        )
                        noise_values["right"][cx_aux][cy_aux][point[2]].append(
                            get_error(point, cx_aux, cy_aux, camera="right")
                        )
                    else:
                        noise_values["left"][cx_aux][cy_aux][point[2]] = [
                            get_error(point, cx_aux, cy_aux, camera="left")
                        ]
                        noise_values["right"][cx_aux][cy_aux][point[2]] = [
                            get_error(point, cx_aux, cy_aux, camera="right")
                        ]
    # Write dictionary to file
    with open("noise_values.pkl", "wb") as f:
        pickle.dump(noise_values, f)
        f.close()
