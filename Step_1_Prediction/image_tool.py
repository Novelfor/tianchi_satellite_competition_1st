import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
from collections import defaultdict
import yuan_tool


def mask_to_bounds(mask):
    return mask_to_bounds_area(mask)[0]

def mask_to_bounds_area(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    _, contours, his = cv2.findContours(mask, 3, 2)
    rects = []
    areas = []
    for index, contour in enumerate(contours):
        if his[0][index][3] != -1:
            continue
        left, top, width, height = cv2.boundingRect(contour)
        rects.append((int(left), int(top), int(left + width), int(top + height)))
        areas.append(np.sum(mask[top: top + height, left: left + width]))

    rects = np.asarray(rects)
    areas = np.asarray(areas)
    return rects, areas

def mask_to_contours_index(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    _, contours, his = cv2.findContours(mask, 3, 2)
    rects = []
    areas = []
    new_contours = []
    for index, contour in enumerate(contours):
        if his[0][index][3] != -1:
            continue
        left, top, width, height = cv2.boundingRect(contour)
        rects.append((int(left), int(top), int(left + width), int(top + height)))
        areas.append(np.sum(mask[top: top + height, left: left + width]))
        new_contours.append(contour)

    rects = np.asarray(rects)
    areas = np.asarray(areas)
    contours = np.asarray(new_contours)
    return rects, areas, contours

def mask_bounds_all_true(mask):
    bounds = mask_to_bounds(mask)
    for bound in bounds:
        left, top, right, bottom = bound
        for i in range(top, bottom):
            for j in range(left, right):
                mask[i, j] = 1
    return mask

def mask_remove_small(mask, threshold):
    mask = np.copy(mask)
    rects, areas, contours = mask_to_contours_index(mask)
    pb = yuan_tool.ProcessBar()
    pb.reset(len(rects))
    for index, rect in enumerate(rects):
        pb.show(index)
        if areas[index] < threshold:
            left, top, right, bottom = rect
            for i in range(top, bottom):
                for j in range(left, right):
                    if cv2.pointPolygonTest(contours[index], (j, i), False) >= 0:
                        mask[i, j] = 0
    return mask

def mask_remove_small_height_width(mask, height_threshold, width_threshold):
    mask = np.copy(mask)
    rects, areas, contours = mask_to_contours_index(mask)
    pb = yuan_tool.ProcessBar()
    pb.reset(len(rects))
    for index, rect in enumerate(rects):
        pb.show(index)
        left, top, right, bottom = rect
        if bottom - top < height_threshold or right - left < width_threshold:
            for i in range(top, bottom):
                for j in range(left, right):
                    if cv2.pointPolygonTest(contours[index], (j, i), False) >= 0:
                        mask[i, j] = 0
    return mask

