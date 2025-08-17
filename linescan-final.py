#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math
import geopandas as gpd
from shapely.geometry import Point
import concurrent.futures
import sys
from shapely.geometry import LineString,Polygon,MultiPoint
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon



# In[8]:


def process_shapefiles(target_gdf, point, buffer_distance):
    # 确保点Shapefile的坐标系与目标Shapefile的坐标系相同
     # 创建缓冲区
    buffer_gdf = point.geometry.buffer(buffer_distance) 
     # 使用overlay函数进行裁剪，使用intersection类型
    #clipped_gdf = gpd.overlay(target_gdf, buffer_gdf, how='intersection')
    clipped_gdf = gpd.clip(target_gdf, buffer_gdf)
    trees_gdf = clipped_gdf[clipped_gdf['class_new'] == '9']
    buildings_gdf = clipped_gdf[clipped_gdf['class_new'] == '7']
    total_area = buffer_gdf.area

    #output_path = "D:\\greenveiw20240821\\case\\trees_gdf.shp"
    #trees_gdf.to_file(output_path, driver='ESRI Shapefile')
    return clipped_gdf,total_area,trees_gdf,buildings_gdf

#Percent Cover of Buildings or Trees
def calculate_percent_cover(trees_gdf,buildings_gdf, total_area):
    total_area_trees = trees_gdf.geometry.area.sum()

    total_area_buildings = buildings_gdf.geometry.area.sum()
    percent_tree = ( total_area_trees/ total_area) * 100
    percent_buildings  = ( total_area_buildings / total_area) * 100

    return percent_tree,percent_buildings


#Mean Height of Buildings or Trees
def calculate_mean_height(clipped_gdf, height_column='height'):
    mean_height = clipped_gdf[height_column].mean()
    return mean_height
#Mean Patch Size of Buildings or Trees
def calculate_mean_patch_size(trees_gdf,buildings_gdf):
    mean_patch_tree = trees_gdf.geometry.area.mean()
    mean_patch_buildings = buildings_gdf.geometry.area.mean()
    return mean_patch_tree,mean_patch_buildings

#Largest Patch Index of Buildings or Trees
def calculate_largest_patch_index(trees_gdf,buildings_gdf, total_area):
    largest_tree_area = trees_gdf.geometry.area.max()
    largest_tree_index = (largest_tree_area / total_area) * 100
    largest_buildings_area = buildings_gdf.geometry.area.max()
    largest_buildings_index = (largest_buildings_area / total_area) * 100
    return largest_tree_index,largest_buildings_index

#Total Edge of Buildings or Trees
def calculate_total_edge(trees_gdf,buildings_gdf):
    total_tree_length = trees_gdf.geometry.length.sum()
    total_buildings_length = buildings_gdf.geometry.length.sum()
    return total_tree_length,total_buildings_length
#Edge Density of Buildings or Trees
def calculate_edge_density(total_tree_length,total_buildings_length, total_area):
    tree_density = total_tree_length / total_area
    buildings_density = total_buildings_length / total_area
    return tree_density,buildings_density

def generate_lines(original_x, original_y, buffer_distance, num_angles, beijing_crs):
    """
    生成从原点到指定半径的多条视线，并存储为 GeoDataFrame。

    参数:
    original_x (float): 视线的起点 x 坐标
    original_y (float): 视线的起点 y 坐标
    radius (float): 视线的半径
    num_angles (int): 角度的总数（步长为 0.1 度，通常为 3600 表示 3600 个视线）
    target_crs (str): CRS (坐标参考系)，用于 GeoDataFrame

    返回:
    lines_gdf (GeoDataFrame): 包含所有生成视线的 GeoDataFrame
    """
    
    # 生成角度范围，按指定的步长生成
    angles = np.arange(0, num_angles, 1)
    
    # 存储生成的视线
    lines = []

    # 计算每个角度的视线终点，并生成视线
    for angle in angles:
        # 角度转弧度
        angle_rad = np.radians(angle)  # 0.1 度角步长，调整角度
        
        # 计算终点坐标
        end_x = original_x + buffer_distance * np.cos(angle_rad)
        end_y = original_y + buffer_distance* np.sin(angle_rad)
        
        # 创建视线，从中心到终点
        line = LineString([(original_x, original_y), (end_x, end_y)])
        
        # 将生成的视线存储起来
        lines.append(line)

    # 将所有视线存储为 GeoDataFrame，方便后续处理
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=beijing_crs)
    lines_gdf['order'] = lines_gdf.index  # 添加索引列作为顺序标识

    return lines_gdf
def convert_multipolygon_to_lines(gdf):
    """
    将 GeoDataFrame 中的 MultiPolygon 拆解为单独的 Polygon，并将每个 Polygon 转换为边界线。
    
    参数:
        gdf (GeoDataFrame): 包含 Polygon 和 MultiPolygon 的 GeoDataFrame。
        
    返回:
        GeoDataFrame: 所有 Polygon 和 MultiPolygon 边界线的新的 GeoDataFrame。
    """
    # 创建一个新的列表，用于存储拆解后的线条
    lines = []
    
    # 遍历每个几何对象
    for idx, row in gdf.iterrows():
        geometry = row['geometry']
        
        # 如果几何对象是 MultiPolygon，则使用 .geoms 属性来迭代其中的每个 Polygon
        if isinstance(geometry, MultiPolygon):
            for poly in geometry.geoms:
                line = poly.boundary  # 将每个 Polygon 转为边界线
                new_row = row.copy()  # 复制当前行数据，确保保留原始属性
                new_row['geometry'] = line  # 更新几何为线
                lines.append(new_row)  # 添加到结果列表中
        elif isinstance(geometry, Polygon):
            # 如果几何是单个 Polygon，则直接转换为边界线
            row['geometry'] = geometry.boundary
            lines.append(row)

    # 将结果转换为新的 GeoDataFrame
    line_gdf = gpd.GeoDataFrame(lines, columns=gdf.columns)
    
    return line_gdf



def convert_3d_to_2d(geometry):
    """
    将 3D 几何对象转换为 2D，而不改变原始的面结构。
    """
    if geometry.is_empty:
        return geometry
    elif geometry.geom_type == 'Polygon':
        # 保留 x, y 坐标，不转换为边界
        return Polygon([(x, y) for x, y, z in geometry.exterior.coords])
    elif geometry.geom_type == 'MultiPolygon':
        # 对于 MultiPolygon，使用 .geoms 属性访问其中的 Polygon
        return MultiPolygon([Polygon([(x, y) for x, y, z in poly.exterior.coords]) for poly in geometry.geoms])
    elif geometry.geom_type == 'LineString':
        return LineString([(x, y) for x, y, z in geometry.coords])
    else:
        return geometry  # 对于其他类型的几何，不做任何改变




def write_output(output, fid, greenview, buildview, blueview, skyview, openness_ratio, nearest_all, nearest_tree, nearest_building, greenview_nobuild):
    # 生成表头
    headers = ["fid", "Greenview", "Buildview", "Blueview", "Skyview", "openness_ratio", "Nearest_All", "Nearest_Tree", "Nearest_Building", "Greenview_NoBuilding"]
    
    # 生成对应的数据行
    data = [fid, greenview, buildview, blueview, skyview, openness_ratio, nearest_all, nearest_tree, nearest_building, greenview_nobuild]
    
    # 检查文件是否已经存在（追加模式下）
    try:
        with open(output, 'r') as file:
            header_written = True  # 如果文件存在，则表头已写入
    except FileNotFoundError:
        header_written = False  # 如果文件不存在，则需要写入表头

    # 写入数据
    with open(output, 'a') as file:
        if not header_written:
            file.write(",".join(headers) + "\n")  # 写入表头
        file.write(",".join(map(str, data)) + "\n")  # 写入数据行



# In[13]:


# In[9]:


points_file = sys.argv[1]
target_file = sys.argv[2]
output = sys.argv[3]
# points_file = r"D:\北京市5环\10 (2).shp"

# target_file =r"D:\北京市5环\10 (1).shp"
# output = r"D:\北京市5环\greenview10.txt"
points_gdf = gpd.read_file(points_file)
target_gdf = gpd.read_file(target_file)
#output = 'D:\\greenveiw20240821\\case\\greenview1.txt'
beijing_crs = 'EPSG:32650'
    # 将 points_gdf 和 target_gdf 转换为适合北京的投影坐标系
points_gdf = points_gdf.to_crs(beijing_crs)
target_gdf = target_gdf.to_crs(beijing_crs)
landcape_index=[]
result = []
for fidname, row in points_gdf.iterrows():
    point1 = row
    original_x = point1.geometry.x
    original_y = point1.geometry.y
    original_z = 1.6
    buffer_distance = 100


    #clipped_gdf.to_file("D:\\greenveiw20240821\\数据对比\\clip_gdf.shp", encoding='utf-8')
    num_angles= 360 
    lines_gdf = generate_lines(original_x, original_y, buffer_distance, num_angles, beijing_crs)
    #lines_gdf.to_file("D:\\greenveiw20240821\\数据对比\\lines_gdf.shp", encoding='utf-8')
    #clipped_gdf['geometry'] = clipped_gdf['geometry'].apply(convert_3d_to_2d)
    #clipped_gdf.to_file("D:\\greenveiw20240821\\数据对比\\clipped_gdf.shp", encoding='utf-8')
    # 强制将所有 Polygon 和 MultiPolygon 转为 LineString
# 应用转换函数
    clipped_gdf,total_area,trees_gdf,buildings_gdf= process_shapefiles(target_gdf, point1, buffer_distance)

    clipped_gdf = convert_multipolygon_to_lines(clipped_gdf)
    if clipped_gdf.crs is None:
        # 假设它原本是 EPSG:32650（请确认实际情况）
        clipped_gdf.set_crs(epsg=32650, inplace=True)  # 设置原始坐标系
    
    # 然后再转换为 beijing_crs
    clipped_gdf = clipped_gdf.to_crs(beijing_crs)
    #clipped_gdf['geometry'] = clipped_gdf.boundary
    #clipped_gdf1.to_file("D:\\greenveiw20240821\\数据对比\\clipped1_gdf.shp", encoding='utf-8')
    # 将三维数据转换为二维
    intersection_gdf = gpd.overlay(lines_gdf, clipped_gdf, how='intersection', keep_geom_type=False)
    geometries = []
    attributes = []
    # 3. 遍历每个几何对象，并拆解 MultiPoint 为单个 Point，同时保留属性
    for idx, row in intersection_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, (MultiPoint, Point)):
            # 如果是 MultiPoint，将其拆分成多个 Point，如果是 Point，直接处理
            for point in geom.geoms if isinstance(geom, MultiPoint) else [geom]:
                geometries.append(point)
                attributes.append(row.drop('geometry'))  # 保留其他所有属性
    #加入原点        
    point1_geom_2d = Point(point1.geometry.x, point1.geometry.y)
    # 构造 GeoDataFrame
    point1_gdf = gpd.GeoDataFrame(geometry=[point1_geom_2d], crs=clipped_gdf.crs)

    # 空间叠加分析
    joinedpoint = gpd.sjoin(point1_gdf, target_gdf, how="left", predicate="intersects")
    joinedpoint['height'] = original_z

    intersection_points = gpd.GeoDataFrame(attributes, geometry=geometries, crs=target_gdf .crs)
    if not joinedpoint.empty:
        # 提取 joined 的第一行属性（排除 geometry）
        joined_attrs = joinedpoint.iloc[0].drop(labels='geometry').to_dict()

        # 获取当前 intersection_points 中所有唯一的 order 值
        unique_orders = intersection_points['order'].dropna().unique()

        # 遍历每个 order，添加一行
        new_rows = []
        for o in unique_orders:
            new_row = {col: None for col in intersection_points.columns}
            new_row.update(joined_attrs)
            new_row['geometry'] = Point(original_x, original_y)
            new_row['order'] = o  # 指定唯一的 order 值
            new_rows.append(new_row)

        # 补齐缺失字段
        for col in new_rows[0].keys():
            if col not in intersection_points.columns:
                intersection_points[col] = None

        # 拼接所有新行
        new_rows_gdf = gpd.GeoDataFrame(new_rows, geometry='geometry', crs=intersection_points.crs)
        intersection_points = pd.concat([intersection_points, new_rows_gdf], ignore_index=True)
        
    intersection_points['order1'] = range(len(intersection_points))
    center_point = Point(original_x , original_y)
    intersection_points['distance_to_center'] = intersection_points.geometry.apply(lambda point: point.distance(center_point))
    intersection_points['height'] = pd.to_numeric(intersection_points['height'], errors='coerce')
    #intersection_points['height2'] = pd.to_numeric(intersection_points['height2'], errors='coerce')
    intersection_points_orig = intersection_points.copy()
    
    intersection_points['start']= -original_z *buffer_distance/intersection_points['distance_to_center']
    intersection_points['end']= (intersection_points['height']-original_z)*buffer_distance/intersection_points['distance_to_center']
    intersection_points['start'].replace(-np.inf, 0, inplace=True)
    # 将 NaN 的值替换为 0（end列）
    intersection_points['end'].fillna(0, inplace=True)
    intersection_points= intersection_points.sort_values(by='distance_to_center', ascending=False) 
    unique_categories = intersection_points['order'].unique()

    greenview = 0
    buildview = 0
    blueview = 0
    skyview = 0
    unobstructed_count = 0
    nearest_all = []
    nearest_tree = []
    nearest_building = []
    
    for a in unique_categories:
        order_point = intersection_points[intersection_points['order']==a]
        order_point = order_point.reset_index(drop=True)
        coverage_array = np.zeros(buffer_distance*100)
#考虑树木的连贯性
        results = []
        # 标记所有植被的索引
        nine_indices = order_point[order_point['class_new'].isin([ '2'])].index

        # 初始化变量
        for start_index, end_index in zip(nine_indices[::2], nine_indices[1::2]):
            start_row = order_point.loc[start_index]
            end_row = order_point.loc[end_index]
        
            results.append({
                'order': a,
                "start": start_row['start'],
                "end": end_row['start'],
                "class_new": 2,
                "distance_to_center": (end_row['distance_to_center'] - start_row['distance_to_center']) / 2 + start_row['distance_to_center']
            })


        # 将结果转为DataFrame方便查看
        results_df = pd.DataFrame(results)

        merged_df = pd.concat([order_point , results_df], ignore_index=True)

            # 根据 distance_to_center 列进行排序
        merged_df = merged_df.sort_values(by="distance_to_center",ascending=False).reset_index(drop=True)

      # 考虑植被的连贯性

        for i, row in merged_df.iterrows():

            start = int(row['start']*100)+buffer_distance*50
            end = int(row['end']*100)+buffer_distance*50
            distance = row['distance_to_center']

        # 在矩阵上从start到end赋值为distance，代表该区间被覆盖
            if start >=end:
                coverage_array[end:start] = row['class_new']
            else:
                coverage_array[start:end] = row['class_new']
                
        if not order_point.empty:
        # 最近任意地物
            order_point = intersection_points_orig[intersection_points_orig['order'] == a].reset_index(drop=True)

            # ✅ 最近建筑或树木距离
            order_point_sorted = order_point.sort_values(by='distance_to_center')
            for idx, row in order_point_sorted.iterrows():
                if row['class_new'] in ['7', '9']:
                    nearest_all.append(row['distance_to_center'])
                    break
            # 最近树木
            tree_points = order_point[order_point['class_new'] == '9']
            if not tree_points.empty:
                nearest_tree.append(tree_points['distance_to_center'].min())
            # 最近建筑
            building_points = order_point[order_point['class_new'] == '7']
            if not building_points.empty:
                nearest_building.append(building_points['distance_to_center'].min())

        
        if np.all(coverage_array == 0):
            unobstructed_count += 1  # ⬅️ 没有任何遮挡的方向，视为“开阔”
        green = np.array(coverage_array == 9).sum()/(buffer_distance*100)
        grass = np.array(coverage_array == 2).sum()/(buffer_distance*100)
        build = np.array(coverage_array == 7).sum()/(buffer_distance*100)
        blue = np.array(coverage_array == 6).sum()/(buffer_distance*100)
        
        greenview = greenview+green + grass
        buildview = buildview + build
        blueview = blueview +blue

    #df_result.to_csv('D:\\greenveiw20240821\\case\\greenview15.txt')
    greenview = greenview/num_angles
    buildview = buildview/num_angles
    blueview = blueview/num_angles
    skyview = 1-greenview-buildview- blueview 
    openness_ratio = unobstructed_count / num_angles 
    mean_nearest_all = np.mean(nearest_all) if nearest_all else np.nan
    mean_nearest_tree = np.mean(nearest_tree) if nearest_tree else np.nan
    mean_nearest_building = np.mean(nearest_building) if nearest_building else np.nan
    
    # =========👇 无建筑版本 =========
    intersection_points_nobuild = intersection_points_orig.copy()
    intersection_points_nobuild.loc[intersection_points_nobuild['class_new'] == '7', 'height'] = 0
    intersection_points_nobuild['start'] = -original_z * buffer_distance / intersection_points_nobuild['distance_to_center']
    intersection_points_nobuild['end'] = (intersection_points_nobuild['height'] - original_z) * buffer_distance / intersection_points_nobuild['distance_to_center']
    intersection_points_nobuild['start'].replace(-np.inf, 0, inplace=True)
    intersection_points_nobuild['end'].fillna(0, inplace=True)
    intersection_points_nobuild = intersection_points_nobuild.sort_values(by='distance_to_center', ascending=False)
    greenview_nobuild = 0

    for a in unique_categories:
        order_point = intersection_points_nobuild[intersection_points_nobuild['order'] == a].reset_index(drop=True)
        coverage_array = np.zeros(buffer_distance * 100)
        results = []
        nine_indices = order_point[order_point['class_new'].isin([ '2'])].index
        for start_index, end_index in zip(nine_indices[::2], nine_indices[1::2]):
            start_row = order_point.loc[start_index]
            end_row = order_point.loc[end_index]
            results.append({
                'order': a,
                "start": start_row['start'],
                "end": end_row['start'],
                "class_new": 2,
                "distance_to_center": (end_row['distance_to_center'] - start_row['distance_to_center']) / 2 + start_row['distance_to_center']
            })
        merged_df = pd.concat([order_point, pd.DataFrame(results)], ignore_index=True)
        merged_df = merged_df.sort_values(by="distance_to_center", ascending=False).reset_index(drop=True)
        for i, row in merged_df.iterrows():
            start = int(row['start'] * 100) + buffer_distance * 50
            end = int(row['end'] * 100) + buffer_distance * 50
            if start >= end:
                coverage_array[end:start] = row['class_new']
            else:
                coverage_array[start:end] = row['class_new']
        green = np.array(coverage_array == 9).sum() / (buffer_distance * 100)
        grass = np.array(coverage_array == 2).sum() / (buffer_distance * 100)
        greenview_nobuild += green + grass

    greenview_nobuild = greenview_nobuild / num_angles
    

    # 添加结果
    result.append({
        'order': fidname,
        'greenview': greenview,
        'buildview': buildview,
        'blueview': blueview,
        'skyview': skyview,
        'openness_ratio': openness_ratio,
        'nearest_all': mean_nearest_all,
        'nearest_tree': mean_nearest_tree,
        'nearest_building': mean_nearest_building,
        'greenview_nobuild': greenview_nobuild
    })



    # 写入输出文件
    write_output(
        output,
        fidname,
        greenview,
        buildview,
        blueview,
        skyview,
        openness_ratio,
        mean_nearest_all,
        mean_nearest_tree,
        mean_nearest_building,
        greenview_nobuild
    )

#df_result = pd.DataFrame(result)
#df_result.to_csv('D:\\深圳绿视率项目数据\\shiyan.txt')


# In[ ]:




