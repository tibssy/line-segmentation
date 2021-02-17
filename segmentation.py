import numpy as np
import cv2


def find_Y(crds, column):
    median = np.median(crds[:, 3] - crds[:, 1])
    ys = np.sort(crds[:, column])
    lines = np.where(np.diff(ys) >= median)[0]

    if column == 1:
        lines += 1
        ys = np.append(ys[0], ys[lines])
    else:
        ys = np.append(ys[lines], ys[-1])

    return ys


def pair_Y(yB, yS):
    for y, itr in zip(yB, range(len(yB))):
        acc = yS[yS < y]
        if acc.size != 0:
            acc = acc[-1]
            yB[itr] = acc if abs(y - acc) < h_median / 2 else y

    return  np.abs(yB)

def find_X(st_m):
    out = []
    sorted = st_m[np.argsort(st_m[:, 1])]
    for line_Ys in zipped:
        slice = sorted[np.logical_and(sorted[:, 1] >= line_Ys[0], sorted[:, 1] < line_Ys[1])]
        line = [np.min(slice[:, 0]), line_Ys[0], np.max(slice[:, 2]), line_Ys[1]]
        out.append(line)

    return np.asarray(out)

def data_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    stats = cv2.connectedComponentsWithStats(thresh, connectivity=8)[2][1:][:, :4]
    return stats[np.argsort(stats[:, 0])]


img = cv2.imread('sample.png')
stats = data_from_image(img)
h_median = np.median(stats[:, 3])
stats_mod = np.copy(stats)

stats_mod[:, 2] = stats_mod[:, 0] + stats_mod[:, 2]
stats_mod[:, 3] = stats_mod[:, 1] + stats_mod[:, 3]

multiplier = 0.6
while multiplier <= 2:
    big = stats_mod[stats[:, 3] >= h_median * multiplier]
    small = stats_mod[stats[:, 3] < h_median * multiplier]
    y1_Big = find_Y(big, 1)
    y2_Big = find_Y(big, 3)
    if y1_Big.shape[0] == y2_Big.shape[0]:
        h_median *= multiplier
        break
    else:
        multiplier += 0.1


y1_Small = find_Y(small, 1)
y2_Small = find_Y(small, 3)

paired_y1 = pair_Y(y1_Big, y1_Small)
paired_y2 = pair_Y(y2_Big * -1, y2_Small[::-1] * -1)

zipped = np.vstack((paired_y1, paired_y2)).transpose()
res = find_X(stats_mod)
print("{} lines with {} characters.".format(res.shape[0], big.shape[0]))

for stat in res:
    x1, y1, x2, y2 = stat
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow('img', img)
cv2.waitKey(0)
