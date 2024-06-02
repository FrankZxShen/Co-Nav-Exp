Perception_System_Prompt = """
Given the following information format:

Target of navigation: format-[Name]

Scene object (object detection): format-[Name, Predicted Probability, Coordinates of the Bounding Box (The Upper left and lower right corners) <(x1, y1), (x2, y2)>]

Scene object (semantic segmentation): format-[Name, Boundary Coordinates defined by vertices in clockwise order <(x1, y1), (x2, y2)...>]

You need to determine the reasonableness of the existence of the above scene objects based on the image information. Your goal is to determine if the scene image is worth exploring by the robot. Consider the following points when making your decision:
(1) Examine the relationship between the target object and the scene. If the object detection model predicts the presence of the target object with high probability (e.g., >85%), it indicates that the scene is worth exploring.
(2) Analyze the image and consider the context of the scene, such as ceilings, walls, floors, and windows. Use your knowledge of typical object locations (e.g., beds in bedrooms, TVs in living rooms) to assess the likelihood of finding the target object.
(3) Consider whether the object target the robot is trying to find is close to the image scene, and make sure that the judgment you output does not violate rule (1). For example, a bathtub is unlikely to be found in a bedroom. However, the presence of a door in the image might suggest that the target object could be located behind it, making the scene worth exploring.
(4) Disregard objects that are commonly found in multiple rooms, such as light switches and doors, as they do not provide strong evidence for the presence of the target object.
Your output should be a simple "[Yes, No]" statement indicating whether the scene is worth exploring based on the given criteria.

Now we begin:
"""

# Example:

# Target of navigation: 'bed'

# Scene object (object detection): 
# bench: 0.45, <(186, 325,), (587, 479)>
# bed: 0.65, <(466, 448), (640, 480)>
# dining table: 0.77, <(125, 244), (201, 325)>
# chair: 0.77, <(65, 231), (146, 328)> 
# couch: 0.87, <(169, 247) (640, 475)>

# Scene object (semantic segmentation):
# sofa: <(294, 202), (267, 216), (284, 217), (288, 258), (307, 235)>
# table: <(200, 150), (230, 150), (230, 170), (200, 170)> 

# Are you confident that the robot is worth exploring in this scenario? Please answer Yes or No.

def form_prompt_for_VLM(target, objs, agents_seg_list) -> str:

    object_detection = ''
    for item in objs:
        name, confidence, coords = item
        coord_pairs = coords[0].split(',')
        coord1 = coord_pairs[0], coord_pairs[1]
        coord2 = coord_pairs[2], coord_pairs[3]
        detection = f"{name}: {confidence}, <({int(float(coord1[0]))}, {int(float(coord1[1]))}), ({int(float(coord2[0]))}, {int(float(coord2[1]))})>\n"
        object_detection += detection

    semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in agents_seg_list.items()]) + "\n"

    Perception_Template = """
Target of navigation: {TARGET}

Scene object (object detection): 
{OBJECT_DETECTION}
Scene object (semantic segmentation):
{SEMANTIC_SEGMENTATION}
Are you confident that the robot is worth exploring in this scenario? You only need to answer "Yes" or "No".
"""
    User_Prompt = Perception_Template.format(
                        TARGET = target,
                        OBJECT_DETECTION = object_detection,
                        SEMANTIC_SEGMENTATION = semantic_segmentation
                    )
    User_Prompt = Perception_System_Prompt + User_Prompt



    return User_Prompt

def contains_yes_or_no(VLM_Pred: str) -> str:
    """
    检查模型B的输出字符串是否包含 'Yes' 或 'No'。

    参数:
    - model_b_output: 字符串 模型B的输出

    返回:
    - 如果包含 "Yes"，返回 "Yes"；如果包含 "No"，返回 "No"；如果都不包含，返回 "Neither"
    """
    if "Yes" in VLM_Pred:
        return "Yes"
    elif "No" in VLM_Pred:
        return "No"
    else:
        return "Neither"

def Perception_weight_decision(VLM_Rel: list, VLM_Pred: str) -> str:
    """
    根据模型A的输出权重和模型B的输出 进行加权决策 输出最终的决策结果。
    
    参数:
    - model_a_output: numpy 数组 模型A的输出 形如 [x, y]，其中 x 是 "Yes" 的权重 y 是 "No" 的权重。
    - model_b_output: 字符串 模型B的输出 只可能是 "Yes" 或 "No"。
    
    返回:
    - 最终加权后的决策结果，字符串，"Yes" 或 "No"。
    """
    b_decision = contains_yes_or_no(VLM_Pred)
    if b_decision == "Neither":
        return b_decision
    
    # 从模型A的输出中提取权重
    x, y = VLM_Rel
    
    if b_decision == "Yes":
        weighted_yes_prob = x
        weighted_no_prob = y * (1 - x)
    else:  # b_decision == "No"
        weighted_yes_prob = x * (1 - y)
        weighted_no_prob = y
    
    # 对权重进行归一化处理
    total_prob = weighted_yes_prob + weighted_no_prob
    if total_prob == 0:  # 避免划分零的问题
        return b_decision
    
    weighted_yes_prob /= total_prob
    weighted_no_prob /= total_prob
    
    # 最终决策
    return weighted_yes_prob, weighted_no_prob
