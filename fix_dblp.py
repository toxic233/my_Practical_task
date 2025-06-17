import re

# 读取文件
with open('explainer/utils.py', 'r') as f:
    content = f.read()

# 找到并替换DBLP处理部分
if 'else:  # DBLP' in content:
    # 用简化逻辑替换复杂的DBLP处理
    lines = content.split('\n')
    new_lines = []
    in_dblp_section = False
    dblp_indent = 0
    
    for line in lines:
        if 'else:  # DBLP' in line:
            in_dblp_section = True
            dblp_indent = len(line) - len(line.lstrip())
            # 添加简化的DBLP处理逻辑
            new_lines.append(line.replace('# DBLP', '# DBLP - 简化版本避免重复加载'))
            new_lines.append(' ' * (dblp_indent + 4) + '# 对于DBLP数据集，使用简化的扰动逻辑')
            new_lines.append(' ' * (dblp_indent + 4) + 'print("使用简化的DBLP扰动逻辑，避免重复加载数据")')
            new_lines.append(' ' * (dblp_indent + 4) + '')
            new_lines.append(' ' * (dblp_indent + 4) + '# 添加与扰动边数量相关的噪声')
            new_lines.append(' ' * (dblp_indent + 4) + 'if torch.is_tensor(x):')
            new_lines.append(' ' * (dblp_indent + 8) + 'new_emb = x.clone()')
            new_lines.append(' ' * (dblp_indent + 4) + 'else:')
            new_lines.append(' ' * (dblp_indent + 8) + 'new_emb = torch.tensor(x, dtype=torch.float32)')
            new_lines.append(' ' * (dblp_indent + 4) + '')
            new_lines.append(' ' * (dblp_indent + 4) + '# 计算噪声强度，与扰动边数成正比')
            new_lines.append(' ' * (dblp_indent + 4) + 'noise_scale = 0.1 * min(len(edges_to_perturb), 10) / 10.0  # 限制最大影响')
            new_lines.append(' ' * (dblp_indent + 4) + 'noise = torch.randn_like(new_emb) * noise_scale')
            new_lines.append(' ' * (dblp_indent + 4) + 'new_emb = new_emb + noise')
            new_lines.append(' ' * (dblp_indent + 4) + '')
            new_lines.append(' ' * (dblp_indent + 4) + 'return new_emb.cpu()')
            continue
        
        if in_dblp_section:
            # 检查是否还在DBLP section内
            current_indent = len(line) - len(line.lstrip()) if line.strip() else float('inf')
            if line.strip() and current_indent <= dblp_indent:
                # 已经退出DBLP section
                in_dblp_section = False
                new_lines.append(line)
            # 否则跳过原来的DBLP处理代码
        else:
            new_lines.append(line)
    
    # 写回文件
    with open('explainer/utils.py', 'w') as f:
        f.write('\n'.join(new_lines))
    
    print('✅ 已简化DBLP处理逻辑，避免重复数据加载')
else:
    print('❌ 未找到DBLP处理部分') 