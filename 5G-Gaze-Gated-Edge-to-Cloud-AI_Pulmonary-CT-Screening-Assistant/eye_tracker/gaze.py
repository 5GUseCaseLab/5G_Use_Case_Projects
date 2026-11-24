import joblib
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError("Invalid gaze direction")
    return v / norm


def train_gaze_model(features, directions):
    """
    features:   shape (N, F), pupil/eye features
    directions: shape (N, 2), target gaze positions in the frontal camera frame
                expressed in the forward-camera coordinate frame.
    """
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1.0),
    )

    model.fit(features, directions)
    return model


def predict_gaze_direction(model, features):
    direction = model.predict(
        np.asarray(features).reshape(1, -1)
    )[0]

    return unit_vector(direction)


def save_gaze_model(model, path="gaze_model.joblib"):
    joblib.dump(model, path)


def load_gaze_model(path="gaze_model.joblib"):
    return joblib.load(path)

def gaze_direction_to_pixel(direction, camera_matrix):
    """
    direction: 3D gaze direction in the forward camera frame.
    camera_matrix: calibrated 3x3 intrinsic matrix.

    Returns a pixel coordinate or None if the ray points
    behind or parallel to the image plane.
    """
    dx, dy, dz = direction

    if dz <= 1e-6:
        return None

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    u = fx * dx / dz + cx
    v = fy * dy / dz + cy

    return int(round(u)), int(round(v))


if __name__ == "__main__":
    # pupil_coords = [
    #     (252, 231), (298, 211), (340, 271), (475, 217), (585, 248),
    #     (582, 265), (499, 248), (445, 229), (428, 214), (276, 283),
    #     (399, 257), (396, 294), (319, 219)
    # ]

    # gaze_coords = [
    #     (284, 235),(308, 225),(330, 255),(410, 228),(519, 244),
    #     (515, 252),(429, 244),(389, 234),(378, 226),(297, 261),
    #     (361, 248),(359, 267),(319, 229)
    # ]
    pupil_coords = []
    gaze_coords = []
    gaze_coordinates = [
        ((567, 194), (639, 0)),
        ((198, 268), (1, 477)),
        ((208, 290), (31, 479)),
        ((216, 215), (0, 0)),
        ((210, 223), (2, 9)),
        ((542, 216), (639, 0)),
        ((212, 225), (3, 0)),
        ((569, 275), (614, 457)),
        ((581, 232), (639, 2)),
        ((248, 276), (13, 479)),
        ((249, 282), (0, 462)),
        ((571, 288), (621, 464)),
        ((182, 201), (0, 0)),
        ((215, 202), (0, 0)),
        ((547, 179), (639, 0)),
        ((541, 220), (596, 0)),
        ((545, 186), (639, 0)),
        ((555, 282), (639, 479)),
        ((568, 266), (625, 479)),
        ((237, 266), (5, 479)),
        ((199, 222), (0, 10)),
        ((579, 282), (639, 479)),
        ((511, 190), (634, 0)),
        ((207, 218), (0, 0)),
        ((191, 255), (0, 476)),
        ((573, 281), (636, 479)),
        ((561, 231), (618, 0)),
        ((192, 200), (0, 0)),
        ((555, 194), (610, 0)),
        ((554, 209), (623, 0)),
        ((569, 312), (614, 479)),
        ((545, 207), (639, 0)),
        ((243, 268), (5, 479)),
        ((219, 267), (5, 465)),
        ((562, 208), (639, 0)),
        ((570, 292), (639, 479)),
        ((205, 199), (0, 14)),
        ((593, 282), (621, 477)),
        ((213, 218), (3, 5)),
        ((202, 204), (0, 6)),
        ((554, 226), (617, 3)),
        ((555, 269), (629, 479)),
        ((566, 262), (634, 469)),
        ((573, 288), (614, 479)),
        ((187, 199), (5, 1)),
        ((201, 202), (4, 0)),
        ((554, 193), (639, 0)),
        ((574, 260), (639, 479)),
        ((539, 215), (634, 3)),
        ((576, 272), (639, 479)),
        ((552, 204), (639, 5)),
        ((564, 216), (639, 33)),
        ((190, 196), (38, 0)),
        ((210, 191), (14, 13)),
        ((583, 293), (639, 479)),
        ((172, 204), (0, 0)),
        ((236, 260), (23, 470)),
        ((191, 201), (0, 0)),
        ((546, 203), (639, 0)),
        ((546, 291), (625, 479)),
        ((590, 279), (635, 463)),
        ((549, 277), (639, 479)),
        ((590, 287), (639, 453)),
        ((210, 253), (8, 479)),
        ((567, 294), (639, 479)),
        ((548, 181), (639, 0)),
        ((222, 207), (0, 0)),
        ((551, 269), (639, 479)),
        ((191, 204), (0, 0)),
        ((246, 265), (0, 478)),
        ((193, 191), (21, 4)),
        ((563, 281), (639, 479)),
        ((571, 273), (639, 479)),
        ((578, 211), (639, 0)),
        ((233, 192), (0, 11)),
        ((518, 188), (612, 0)),
        ((220, 235), (14, 479)),
        ((245, 255), (0, 479)),
        ((220, 287), (0, 449)),
        ((211, 193), (16, 0)),
        ((536, 200), (639, 24)),
        ((572, 284), (639, 479)),
        ((525, 224), (639, 12)),
        ((567, 299), (595, 459)),
        ((218, 193), (17, 5)),
        ((568, 210), (637, 5)),
        ((539, 192), (639, 0)),
        ((566, 271), (632, 479)),
        ((562, 269), (639, 471)),
        ((232, 280), (0, 463)),
        ((189, 207), (4, 19)),
        ((230, 280), (32, 479)),
        ((578, 185), (639, 0)),
        ((240, 269), (0, 452)),
        ((195, 193), (0, 0)),
        ((206, 205), (0, 4)),
        ((549, 209), (639, 0)),
        ((571, 221), (634, 5)),
        ((201, 275), (0, 479)),
        ((565, 267), (637, 457)),
        ((563, 253), (623, 475)),
        ((563, 290), (639, 479)),
        ((564, 304), (636, 479)),
        ((575, 265), (626, 479)),
        ((567, 222), (638, 19)),
        ((567, 289), (639, 474)),
        ((585, 275), (639, 465)),
        ((552, 286), (617, 476)),
        ((574, 280), (626, 476)),
        ((210, 200), (0, 5)),
        ((201, 208), (3, 0)),
        ((191, 201), (16, 0)),
        ((222, 256), (0, 475)),
    ]

    for i in range(len(gaze_coordinates)):
        pupil_coords.append(gaze_coordinates[i][0])
        gaze_coords.append(gaze_coordinates[i][1])


    save_gaze_model(train_gaze_model(pupil_coords, gaze_coords))
