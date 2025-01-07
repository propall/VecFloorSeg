import copy
import os
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
- This code processes floor plan SVGs from the CubiCasa5k dataset
- It performs Delaunay triangulation on floor plans
- It extracts wall shapes, doors, and windows
- It builds dual graph relationships
- Finally, it saves the processed data into pickle files

"""


# import sys
# Add the parent directory of DataPreparation to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataPreparation import WALLTYPE, NOTWALL, DOORTYPE, WINDOWTYPE, PARTITION, NOTPARTITION, \
    TARGET_DIR, REFER_NAME, CLASSES
from PIL import Image
from Utils.svgUtils import PolygonWall, get_points
from Utils import triPlot as trP
from xml.dom.minidom import parse, Node
from skimage.draw import polygon
from Utils.graphicsUtilsRe import isLineIntersection, graphCrune
from Utils.extendWall import extendFloatingWall, extendCornerWall
from matplotlib.colors import ListedColormap


class SVGParser:
    """
    # Base class for parsing SVG files
    # - Extracts shapes like lines, paths, circles
    # - Gets image dimensions
    # - Gets wall shapes
    
    dom = xml.dom.minidom.parse('xml_file.xml')
    root = dom.documentElement
    
    - Parsing the SVG XML file and traversing its structure to extract geometric shapes (lines, paths, circles).
    
    
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.dom = parse(filepath)
        self.root = self.dom.documentElement

        self.shapes = ['line', 'path', 'circle'] # supported SVG elements in the code refer to the different types of SVG XML tags that describe basic shapes.
        self.filtered_nodename = ['image', 'g', 'defs'] # specifies the tags to ignore (<image>: Embedded raster images; <g>: Group tags that wrap several shapes for organization but don’t define actual geometry; <defs>: Definitions (e.g., reusable shapes or styles), not directly part of the SVG structure)

    def _traverse_tree(self, root, ret_list, parent_attrs):
        """
        - Recursively traverses the XML DOM (Document Object Model).
        - When it encounters a supported shape (i.e., a <line>, <path>, or <circle> tag), it extracts the tag’s attributes (coordinates, shape details) and adds them to "ret_list"
        
        """
        parent_attrs = copy.copy(parent_attrs)

        if root.attributes is not None:
            attrs = root.attributes.items()
            for att in attrs:
                parent_attrs[att[0]] = att[1]

        for child in root.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                if child.nodeName in self.shapes:
                    attrs = child.attributes.items()
                    # print(attrs, child.nodeType)
                    attr_dict = copy.copy(parent_attrs)
                    for att in attrs:
                        attr_dict[att[0]] = att[1]
                    attr_dict['shape_name'] = child.nodeName
                    ret_list.append(attr_dict)
                elif child.nodeName not in self.filtered_nodename:
                    print('node is not a supported shape', child.tagName)
                    raise NotImplementedError('in file {}'.format(self.filepath))
            self._traverse_tree(child, ret_list, parent_attrs)

    def get_all_shape(self):
        # Collects and returns all shapes in the SVG document.
        ret_list = []
        self._traverse_tree(self.root, ret_list, {})
        return ret_list

    def get_image_size(self):
        # Extracts the width and height of the image embedded in the SVG file.
        img_info = self.root.getElementsByTagName('image')[0]
        img_width = img_info.getAttribute('width')
        img_height = img_info.getAttribute('height')
        return float(img_width), float(img_height)

    def getWallShape(self):
        # Extracts wall vertices from the SVG paths tagged with "Wall" or "Railing"
        # The code that defines boundaries to create "primal graph" in VecFloorSeg paper
        wallList = []
        wallVertices = {}
        vIdx = 0

        for e in self.dom.getElementsByTagName('g'):
            layerID = e.getAttribute("id")
            if "WALL" in layerID and "TEXT" not in layerID:
                self._traverse_tree(e, wallList, {})
            else:
                pass

        for path in wallList:
            # for p in path['d']:
            #     print(p)
            operations = path['d'].split(' ')
            opIdxes = range(0, len(operations), 2)
            vPre = None
            vStart = None
            for opIdx in opIdxes:
                if operations[opIdx] == "M": # move,
                    xy = operations[opIdx + 1].split(',')
                    v = (float(xy[0]), float(xy[1]))
                    if not wallVertices.__contains__(v):
                        wallVertices[v] = [vIdx]
                        vIdx += 1
                    else:
                        pass
                    vPre = vStart = v

                elif operations[opIdx] == "L": # line
                    xy = operations[opIdx + 1].split(',')
                    v = (float(xy[0]), float(xy[1]))
                    if not wallVertices.__contains__(v):
                        wallVertices[v] = [vIdx]
                        vIdx += 1
                    wallVertices[v].append(wallVertices[vPre][0])
                    vPre = v

                elif operations[opIdx] == 'z':
                    wallVertices[vStart].append(wallVertices[vPre][0])

                else:
                    raise NotImplementedError(
                        'in {}, wall has another operation {}'.
                            format(self.filepath, operations[opIdx])
                    )
        return wallVertices


class SVGParserCUBI(SVGParser):
    """
    This is a subclass of SVGParser and processes SVGs in the specific format used in the Cubicasa5k dataset.
    
    In SVG images, different parts of the floor plan (walls, doors, windows) are grouped under group tags(<g id=''>) with specific id attributes.
    <g id="Wall">: Contains wall shapes.
    <g id="Door">: Contains door shapes.
    <g id="Window">: Contains window shapes.
    
    # Specialized parser for CubiCasa-5k dataset
    # - Inherits from SVGParser
    # - Processes walls, doors, windows
    # - Maps elements to vertices
    """
    def __init__(self, filepath):
        super().__init__(filepath)
        self.wallIdx = 1

        imgPth = filepath.replace('model.svg', REFER_NAME)
        img = Image.open(imgPth)
        height, width = img.height, img.width
        self.shape = (height - 1, width - 1)

    def getWallShape(self):
        """
        :return:
            wallVertices : dict(key=coord, value=[lines endpoints id])
            doors: list(door primitive idx)
            windows: list(window primitive idx)
        """
        wallVertices = {} # Dictionary to store vertices of walls.
        
        """
        Triangulation Holes:
        - In the context of Delaunay triangulation and floor plan segmentation, a triangulation hole refers to a region within the triangulation process that is excluded from being filled with triangles. 
        - These regions are typically represented as polygonal voids or gaps that do not get subdivided into triangles during the triangulation process.
        
        Why are triangulation holes used in floor plan segmentation?
        In the context of floor plans:
        - The triangulation is focused on accurately representing walls and boundaries.
        - The center of the rooms (the floor) may be left out as a "hole" because the floor is a continuous region, not a structural element.
        - If you want to detect the floors separately, you would then use a separate segmentation process to detect the largest contiguous region (the untriangulated floor area).
        """
        
        holes = [] # Center points of walls for triangulation holes.
        primitiveDoors, primitiveWindows = [], [] # Store primitive indices for doors and windows.
        vIdx = 0 # Vertex index counter
        for e in self.dom.getElementsByTagName('g'):
            if e.getAttribute("id") == "Wall" or e.getAttribute("id") == "Railing":
                wall = PolygonWall(e, self.wallIdx, self.shape)
                holes.append(list(wall.center))

                for i in range(4):
                    v = (wall.X[i], wall.Y[i])
                    vPre = (wall.X[i - 1], wall.Y[i - 1])
                    if not wallVertices.__contains__(vPre):
                        wallVertices[vPre] = [vIdx]
                        vIdx += 1

                    if not wallVertices.__contains__(v):
                        wallVertices[v] = [vIdx, wallVertices[vPre][0]]
                        vIdx += 1
                    else:
                        wallVertices[v].append(wallVertices[vPre][0])

            if e.getAttribute("id") == "Window" or e.getAttribute("id") == "Door":

                X, Y = get_points(e)
                if self.shape:
                    X = np.clip(X, 0, self.shape[1])
                    Y = np.clip(Y, 0, self.shape[0])
                # Y, X = clipBoundary(Y, X, height=self.shape[0], width=self.shape[1])
                primitiveIdxPair = []
                for i in range(4):
                    v = (X[i], Y[i])
                    vPre = (X[i - 1], Y[i - 1])
                    if not wallVertices.__contains__(vPre):
                        wallVertices[vPre] = [vIdx]
                        vIdx += 1

                    if not wallVertices.__contains__(v):
                        wallVertices[v] = [vIdx, wallVertices[vPre][0]]
                        primitiveIdxPair.append(vIdx)
                        vIdx += 1
                    else:
                        wallVertices[v].append(wallVertices[vPre][0])
                        primitiveIdxPair.append(wallVertices[v][0])

                if e.getAttribute("id") == "Window":
                    primitiveDoors.append(primitiveIdxPair)
                else:
                    primitiveWindows.append(primitiveIdxPair)

        return wallVertices, primitiveDoors, primitiveWindows


def delaunayTriangulation(verticesDict, filePth, scaleCoeff=10, **kwargs):
    """
    # Performs triangulation on the floor plan vertices
    # Creates a mesh of triangles from the vertices
    
    :param verticesDict:
        dict(
            key= coordintaes of floorplan endpoints，
            value=list(endpoint id)
        )；
    :param filePth:
    :param scaleCoeff: zoom in coefficient of coordinates
    :param kwargs:
    :return:
    """
    # fileName = filePth.split('\\')[-2]
    fileName = os.path.basename(os.path.dirname(filePth))

    lblImg = Image.open('{}/annotation/{}.png'.format(TARGET_DIR, fileName))

    width, height = lblImg.width, lblImg.height
    w, h = width * 1., height * 1.
    vs = len(verticesDict.keys())
    minX, minY = 0., 0.
    maxX, maxY = w - 1., h - 1.

    pointsBdary = [(minX, minY), (maxX, minY), (maxX, maxY), (minX, maxY)] # xy coord
    pointsBdaryIdx = []
    newVsNum = 0
    for point in pointsBdary:
        if verticesDict.__contains__(point):
            pointsBdaryIdx.append(verticesDict[point][0])
        else:
            verticesDict[point] = [vs + newVsNum]
            pointsBdaryIdx.append(vs + newVsNum)
            newVsNum += 1
    pointsBdaryIdx.append(pointsBdaryIdx[0])
    for idx, point in enumerate(pointsBdary):
        verticesDict[point].append(pointsBdaryIdx[idx + 1])

    vertices, edgeIndex = isLineIntersection(verticesDict, scaleCoeff=scaleCoeff)
    walls = dict(vertices=vertices, segments=edgeIndex, **kwargs)
    if kwargs.__contains__('excludeVIdxes'):
        walls, extendWalls = extendCornerWall(
            walls, min(maxX, maxY), scaleCoeff=scaleCoeff,
            excludeVIdxes=kwargs['excludeVIdxes']
        )
    else:
        walls, extendWalls = extendCornerWall(
            walls, min(maxX, maxY), scaleCoeff=scaleCoeff
        )

    num_extend1_vs = len(walls['vertices'])
    vertices, edgeIndex = isLineIntersection(walls, scaleCoeff=1)
    walls = dict(vertices=vertices, segments=edgeIndex)

    for vIdx in range(num_extend1_vs, len(vertices)):
        for edge in edgeIndex:
            if edge[0] == vIdx or edge[1] == vIdx:
                extendWalls.append(edge)

    op = 'p'
    segWalls = tr.triangulate(walls, op)
    return walls, segWalls, extendWalls


def plotProblemPts(vertices, *pts):

    fig, ax = plt.subplots()
    trP.vertices(ax, vertices=vertices)
    for pt in pts:
        coordPt = vertices[pt]
        ax.scatter(coordPt[0], coordPt[1], color='r', s=1)

    fig.show()
    plt.show()
    plt.close()


def triangleCorrespondingLabel(segWalls, svgPth, scaleCoeff=1):
    """
    # Maps triangles to their corresponding labels in the annotation
    # Helps identify what each triangle represents (wall, room, etc.)
    
    :param segWalls: 进行delaunay三角剖分后的数据结构
    :param svgPth: 数据的名称
    :param scaleCoeff: 缩放比例
    :return: 向segWalls中加入triangles_label和cmap属性,
        shape(triangles_label) == shape(segWalls['triangles'])
    """
    # fileName = svgPth.split('\\')[-2]
    fileName = os.path.basename(os.path.dirname(svgPth))
    lblImg = Image.open('{}/annotation/{}.png'.format(TARGET_DIR, fileName))
    lblArray = np.array(lblImg, dtype=np.uint8)

    # # 初始化ndarray
    # 缩放到原始图像的尺寸
    h, w = lblArray.shape
    originVCoords = segWalls['vertices'] / float(scaleCoeff)
    originVCoords = np.around(originVCoords, 1).astype(int)

    # 计算label
    triangles = segWalls['triangles'].tolist()
    lblTris = np.zeros([len(triangles), 1], dtype=np.uint8)
    areaTris = np.zeros([len(triangles), 1], dtype=float)
    for idx, triangle in enumerate(triangles):
        triX, triY = originVCoords[triangle, 0], originVCoords[triangle, 1]
        area = 0.5 * np.abs(
            (triX[0] * triY[1] + triX[1] * triY[2] + triX[2] * triY[0]) - \
            (triX[1] * triY[0] + triX[2] * triY[1] + triX[0] * triY[2])
        )
        areaTris[idx] = area

        rr, cc = polygon(triY, triX) # shape(rr) == shape(cc)
        rrClip, ccClip = clipBoundary(rr, cc, h, w)
        lblTriCandidate = lblArray[rrClip, ccClip]
        if lblTriCandidate.shape[0] == 0:
            # 这个三角形面积在当前scaleCoeff太小了，将它的label设置为255(ignore)
            temp = 255
        else:
            lblTri, lblTriCounts = np.unique(lblTriCandidate, return_counts=True)
            tempIdx = np.argmax(lblTriCounts)
            temp = lblTri[tempIdx]

        lblTris[idx] = temp

    segWalls['triangles_label'] = lblTris
    segWalls['triangles_area'] = areaTris
    assert 0.95 < np.sum(areaTris) / (h * w) < 1.05, "wrong triangle area calculation."

    palette = lblImg.getpalette()
    palette = np.array(palette).reshape((256, 3)) / 256.
    paletteTransparency = np.ones((256, 1))
    palette = np.concatenate([palette, paletteTransparency], axis=-1)

    segWalls['cmap'] = ListedColormap(palette)
    return segWalls


def plotTriangles(walls, segWalls, filePth, saveDir, plot_label=True):

    trP.compare(plt, walls, segWalls, figsize=(12, 10), plot_label=plot_label)
    ax3 = plt.subplot(236)

    img = plt.imread(os.path.join(filePth, REFER_NAME))
    ax3.imshow(img)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(os.path.join(saveDir, os.path.basename(os.path.dirname(filePth)) + '.png'))
    plt.close()


def plotVenoGraph(topoDict, filePth, saveDir):
    fig, ax = plt.subplots()
    from Utils.triPlot import plot
    plot(ax, **topoDict)
    fig.savefig(os.path.join(saveDir, os.path.basename(os.path.dirname(filePth)) + '_triGraph.png'))
    plt.close()


def plotMergeGraph(
        walls, segWalls, topoDict,
        filePth, saveDir, plot_label=True
):
    from Utils.triPlot import plot
    trP.compare(
        plt, walls, segWalls,
        figsize=(12, 10), plot_label=plot_label
    )
    ax3 = plt.subplot(236)

    img = plt.imread(os.path.join(filePth, REFER_NAME))
    ax3.imshow(img)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)

    ax = plt.subplot(234)
    plot(ax, **topoDict)
    plt.savefig(os.path.join(saveDir, os.path.basename(os.path.dirname(filePth)) + '_merge.png'))
    plt.close()


def clipBoundary(rr, cc, height, width):
    s = np.column_stack((rr, cc))
    s = s[s[:, 0] < height]
    s = s[s[:, 1] < width]
    return s[:, 0], s[:, 1]


def _formAnEdge(pIdx1, pIdx2):
    assert pIdx1 != pIdx2, \
        "wrong topology found in triangle, " \
        "pIdx1={}, pIdx2={}".format(pIdx1, pIdx2)
    return (pIdx1, pIdx2) if pIdx1 < pIdx2 else (pIdx2, pIdx1)


def _genTriangleGraph(nodePos, triangles, walls, extendWalls, pDoors, pWindows):

    def judgeEdgeType(edge_candidate):
        pt1, pt2 = edge_candidate
        pt1_belong, pt2_belong = -1, -1
        for pIdx, p in enumerate(pDoors + pWindows):
            if pt1 in p:
                pt1_belong = pIdx
            if pt2 in p:
                pt2_belong = pIdx
            if pt1_belong != -1 and pt2_belong != -1:
                break

        if pt1_belong == pt2_belong:
            if pt1_belong == -1:
                return WALLTYPE
            elif pt1_belong < len(pDoors):
                return DOORTYPE
            else:
                return WINDOWTYPE
        else:
            if pt1_belong == -1:
                if pt2_belong < len(pDoors):
                    return DOORTYPE
                else:
                    return WINDOWTYPE
            elif pt2_belong == -1:
                if pt1_belong < len(pDoors):
                    return DOORTYPE
                else:
                    return WINDOWTYPE
            else:
                return WALLTYPE


    tEdge, tEdgeAttr, tEdgeType, tEdgeDual = [], [], [], [] # edge between triangles
    node, nodeAttr = [], []
    ETPairs = {}
    for idx, triangle in enumerate(triangles.tolist()):

        points = nodePos[triangle]
        pointsTemp = np.concatenate([points, points[0, :][np.newaxis, :]], axis=0)
        pointsTemp = pointsTemp[1:, :]

        edgeLen = np.linalg.norm(points - pointsTemp, axis=1)
        edgeSum = np.sum(edgeLen)
        x = edgeLen[0] * points[2, 0] + \
            edgeLen[1] * points[0, 0] + \
            edgeLen[2] * points[1, 0]
        y = edgeLen[0] * points[2, 1] + \
            edgeLen[1] * points[0, 1] + \
            edgeLen[2] * points[1, 1]
        node.append([x / edgeSum, y / edgeSum])
        nodeAttr.append([
            *node[-1], *np.reshape(points, -1).tolist()
        ])

        triangle.append(triangle[0])
        for i in range(3):
            edge = _formAnEdge(triangle[i], triangle[i + 1])

            if edge in walls:
                if edge in extendWalls:
                    edgeType = NOTWALL
                else:
                    edgeType = judgeEdgeType(edge)
            else:
                edgeType = NOTWALL
            if ETPairs.__contains__(edge):
                ETPairs[edge].append(idx)
            else:
                ETPairs[edge] = [edgeType, idx]

    for key, value in ETPairs.items():
        if len(value) > 2:
            assert len(value) == 3, \
                "wrong edge topology generated，" \
                "more than two triangles share one edge. Notice!"
            tEdge.append(value[1:])
            tEdgeType.append(value[0])
            tEdgeAttr.append([
                *node[value[1]], *node[value[2]]
            ])
            tEdgeDual.append(key)
        else:
            value.append(-1) # If the value of one point is -1, then the edge does not have a pair of edges
            pass

    return tEdge, tEdgeAttr, tEdgeType, node, nodeAttr, tEdgeDual, ETPairs


def buildDualRelationship(segWalls, venoGraph, edgeTriPairs):
    edgeSet = list(edgeTriPairs.keys())
    hasDualSubset = venoGraph['segment_dual']
    xHasDualSubset = list(
        set(edgeSet).difference(set(hasDualSubset))
    )

    partiSubset = []
    xPartiSubset = []
    lblTris = segWalls['triangles_label']
    for edge in hasDualSubset:
        idx1, idx2 = edgeTriPairs[edge][1], edgeTriPairs[edge][2]
        if lblTris[idx1] == lblTris[idx2]:
            xPartiSubset.append(edge)
        else:
            partiSubset.append(edge)
    xPartiSubset.extend(xHasDualSubset)

    for e in partiSubset:
        edgeTriPairs[e].append(PARTITION)
    for e in xPartiSubset:
        edgeTriPairs[e].append(NOTPARTITION)

    segWalls['edge'] = edgeSet
    segWalls['edge_attr'] = [(t[0], t[3]) for t in edgeTriPairs.values()] # wall, parti
    segWalls['edge_dual'] = [(t[1], t[2]) for t in edgeTriPairs.values()]
    return segWalls


def worker(svgPth, dataVerifyDir=None):
    """
    # Main processing function that:
    # 1. Parses SVG
    # 2. Gets wall shapes
    # 3. Performs triangulation
    # 4. Builds graph relationships
    # 5. Returns processed data
    """
    s = 50

    p = SVGParserCUBI(svgPth + 'model.svg')
    vDict, pDoors, pWindows = p.getWallShape()

    ecldPVs = []
    for p in pDoors + pWindows:
        ecldPVs = ecldPVs + p
    walls, segWalls, extendWalls = delaunayTriangulation(
        vDict, svgPth, scaleCoeff=s, excludeVIdxes=ecldPVs
    )
    segWalls = triangleCorrespondingLabel(segWalls, svgPth, scaleCoeff=s)

    fWall = []
    for wall in segWalls['segments']:
        w = (wall[0], wall[1]) if wall[0] < wall[1] else (wall[1], wall[0])
        fWall.append(w)

    tEdges, tEdgeAttr, tEdgeType, nodes, nodeAttr, tEdgeDual, wholeEdgeSet = \
        _genTriangleGraph(segWalls['vertices'], segWalls['triangles'], fWall, extendWalls, pDoors, pWindows)

    if dataVerifyDir is not None:
        plotTriangles(walls, segWalls, svgPth, dataVerifyDir)

    fplanVeno = {
        'vertices': nodes, 'x': nodeAttr,
        'segments': tEdges, 'segment_attr': tEdgeAttr,
        'segments_type': tEdgeType, 'scale_coeff': s,
        'segment_dual': tEdgeDual
    }
    segWalls = buildDualRelationship(
        segWalls=segWalls, venoGraph=fplanVeno, edgeTriPairs=wholeEdgeSet
    )

    merge_segWalls, merge_fplanVeno = graphCrune(segWalls, fplanVeno, walls, wholeEdgeSet)

    if dataVerifyDir is not None:
        topoDict = {
            'vertices': merge_fplanVeno['vertices'],
            'segments': merge_fplanVeno['segments']
        }
        plotMergeGraph(
            walls, merge_segWalls, topoDict,
            svgPth, dataVerifyDir, plot_label=False
        )

    return os.path.basename(os.path.dirname(svgPth)), segWalls, fplanVeno, merge_segWalls, merge_fplanVeno


if __name__ == "__main__":
    
    

    """CubiCasa Delaunay的批量操作 note the modification of train, test, val"""
    dataDir = "cubicasa5k"
    split = 'test'
    dataFile = "/{}.txt".format(split)
    folders = np.genfromtxt(dataDir + dataFile, dtype='str')

    excludeList = [
        '/high_quality_architectural/2003/',
        '/high_quality_architectural/2565/',
        '/high_quality_architectural/6143/',
        '/high_quality_architectural/10074/',
        '/high_quality_architectural/10754/',
        '/high_quality_architectural/10769/',
        '/high_quality_architectural/14611/',
        '/high_quality/7092/',
        '/high_quality/1692/',

        'high_quality_architectural/10', # img does not match label
    ]
    dataVerificationDir = None
    
    if split == 'val':
        TARGET_DIR = "CUBI_new_val"
    elif split == 'train':
        TARGET_DIR = "CUBI_new_train"  # or whatever your train folder is called
    elif split == 'test':
        TARGET_DIR = "CUBI_new_test"   # or whatever your test folder is called

    ffolders = []
    for x in folders:
        # x = x.replace('/', '\\') # This is Windows specific path, ubuntu can use default
        if x not in excludeList:
            ffolders.append(x)
    fs = [dataDir + x for x in ffolders]
    sorted(fs)
    print("fs: ", fs) # List of folders in dataFile

    """multiple process"""
    import multiprocessing
    from multiprocessing import Pool
    import pickle
    
    # Calculate number of CPU cores to use 
    total_cores = multiprocessing.cpu_count()
    cpuCount = max(3, int(total_cores - (3/8 * total_cores)))  # Convert to int # 3/8th of cores reserved as system cores or 3 cores, whichever is greater
    
    pool = Pool(cpuCount) # Create a process pool
    
    # Initialize dictionaries to store results
    writeIn = {}          # Stores original triangulation and graph data
    merge_writeIn = {}    # Stores merged versions of data
    
    # If data verification directory is provided, create a partial function
    if dataVerificationDir is not None:
        import functools
        func = functools.partial(worker, dataVerifyDir=dataVerificationDir)
    else:
        func = worker
    
    # Parallel Processing
    #pool.imap_unordered: distributes work across processes
    for res in tqdm(pool.imap_unordered(func, fs), total=len(fs)):
        writeIn[res[0]] = [res[1], res[2]]        # Store original results 
        merge_writeIn[res[0]] = [res[3], res[4]]  # Store merged results

    # Save results to pickle files
    with open('{}/{}.pkl'.format(TARGET_DIR, split), 'wb') as f:
        pickle.dump(writeIn, f)
    with open('{}/merge_{}.pkl'.format(TARGET_DIR, split), 'wb') as f:
        pickle.dump(merge_writeIn, f)

    """single process"""
    

    # for f in tqdm(fs[:]):
    #     """
    #     fs[:] creates a shallow copy of the list fs. 
    #     This means all elements are the same, but it ensures that the original list is not modified if changes are made inside the loop.
        
    #     saves visualizations in dataVerificationDir
    #     """
    #     worker(f, dataVerificationDir)
