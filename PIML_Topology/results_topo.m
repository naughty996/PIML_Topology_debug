% 1. 加载数据
data = load('./results_topo/phi_history_all.mat');
phi_all = data.phi_all;
[nx, ny, num_steps] = size(phi_all);

% 2. 配置动画参数
figure('Color', 'w', 'Position', [100, 100, 800, 600]);
v = VideoWriter('./results_topo/topology_evolution.mp4', 'MPEG-4');
v.FrameRate = 15; % 每秒15帧，适合观察演化
open(v);

% 3. 循环渲染每一帧
for i = 1:num_steps
    clf; % 清空当前图形
    
    % 设置灰度底图
    imagesc(phi_all(:, :, i));
    colormap(gray);
    hold on;
    
    % 叠加红色边界线 (phi = 0)
    contour(phi_all(:, :, i), [0 0], 'r', 'LineWidth', 2);
    
    % 添加标注
    title(['演化步数: ', num2str(i)]);
    axis equal tight;
    axis off;
    
    % 写入视频帧
    drawnow;
    frame = getframe(gcf);
    writeVideo(v, frame);
end

close(v);
disp('[*] 动画生成完成: ./results_topo/topology_evolution.mp4');