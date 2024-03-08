import torch
import torch.nn as nn
from seqtr.models import FUSIONS
from seqtr.models.vis_encs.darknet import darknet_conv


@FUSIONS.register_module()
class SimpleFusion(nn.Module):
    def __init__(self,
                 vis_chs=(256, 512, 1024),
                 direction='bottom_up'
                 ):
        super(SimpleFusion, self).__init__()
        self.fp16_enabled = False
        assert direction in ['to_mid', 'bottom_up', 'none']
        self.direction = direction

        if direction == 'bottom_up':
            assert len(vis_chs) == 3
            ch = sum(vis_chs[:2])
            self.down_mid2top = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((ch, ), (ch, ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.top_project = nn.Sequential(
                *darknet_conv((ch+vis_chs[-1], ch+vis_chs[-1], ), (ch+vis_chs[-1], vis_chs[-1], ), (3, 1, ), (1, 1, )))
        elif direction == 'to_mid':
            assert len(vis_chs) == 3
            ch = sum(vis_chs)
            self.up_top2mid = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                *darknet_conv((vis_chs[-1], ), (vis_chs[-1], ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.mid_project = nn.Sequential(
                *darknet_conv((ch, ch, ), (ch, vis_chs[-1], ), (3, 1, ), (3, 1, )))
        elif direction == 'none':
            assert len(vis_chs) == 1

        self.activate = nn.Tanh()

    def forward(self, x, y):
        """Args
            x (list[tensor]): multi-scale feature maps from visual encoders,
                e.g., for darknet, it's [batch_size, 256, 52, 52], [batch_size, 512, 26, 26],
                [batch_size, 1024, 13, 13] in sequential order.

            y (tensor): [batch_size, 1, C_l].

            y_word (tensor): [batch_size, max_token, C_l].

            y_mask (tensor): [batch_size, max_token], dtype=torch.bool, 
                True means ignored position.

        Returns:
            x_multi_modal (tensor): [batch_size, 512, 13, 13]
        """
        if self.direction == 'bottom_up':
            l, m, s = x
            m = torch.cat([self.down_bot2mid(l), m], 1)
            s = torch.cat([self.down_mid2top(m), s], 1)
            x_vis_enc = self.top_project(s)
        elif self.direction == 'to_mid':
            l, m, s = x
            s = self.up_top2mid(s)
            l = self.down_bot2mid(l)
            m = torch.cat([s, m, l], 1)
            x_vis_enc = self.mid_project(m)
        elif self.direction == 'none':
            x_vis_enc = x

        y_2d = y.squeeze().unsqueeze(-1).unsqueeze(-1)
        x_multi_modal = self.activate(x_vis_enc) * self.activate(y_2d)

        return x_multi_modal


@FUSIONS.register_module()
class SimpleFusionv2(nn.Module):
    def __init__(self,
                 vis_chs=(256, 512, 1024),
                 direction='bottom_up',
                 fusion_dim=1024,
                 ):
        super(SimpleFusionv2, self).__init__()
        self.fp16_enabled = False
        assert direction in ['to_mid', 'bottom_up', 'none']
        self.direction = direction

        if direction == 'bottom_up':
            assert len(vis_chs) == 3
            ch = sum(vis_chs[:2])
            self.down_mid2top = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((ch, ), (ch, ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.top_project = nn.Sequential(
                *darknet_conv((ch+vis_chs[-1], ch+vis_chs[-1], ), (ch+vis_chs[-1], vis_chs[-1], ), (3, 1, ), (1, 1, )))
        elif direction == 'to_mid':
            assert len(vis_chs) == 3
            ch = sum(vis_chs)
            self.up_top2mid = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                *darknet_conv((vis_chs[-1], ), (vis_chs[-1], ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.mid_project = nn.Sequential(
                *darknet_conv((ch, ch, ), (ch, vis_chs[-1], ), (3, 1, ), (3, 1, )))
        elif direction == 'none':
            pass

        self.activate = nn.Tanh()
        # self.linear = nn.Conv2d(vis_chs[-1],fusion_dim,1,1)
        self.linear = nn.Sequential(*darknet_conv((vis_chs[-1], ), (fusion_dim, ), (1, ), (1, )))

    def forward(self, x, y):
        """Args
            x (list[tensor]): multi-scale feature maps from visual encoders,
                e.g., for darknet, it's [batch_size, 256, 52, 52], [batch_size, 512, 26, 26],
                [batch_size, 1024, 13, 13] in sequential order.

            y (tensor): [batch_size, 1, C_l].

            y_word (tensor): [batch_size, max_token, C_l].

            y_mask (tensor): [batch_size, max_token], dtype=torch.bool, 
                True means ignored position.

        Returns:
            x_multi_modal (tensor): [batch_size, 512, 13, 13]
        """
        if self.direction == 'bottom_up':
            l, m, s = x
            m = torch.cat([self.down_bot2mid(l), m], 1)
            s = torch.cat([self.down_mid2top(m), s], 1)
            x_vis_enc = self.top_project(s)
        elif self.direction == 'to_mid':
            l, m, s = x
            s = self.up_top2mid(s)
            l = self.down_bot2mid(l)
            m = torch.cat([s, m, l], 1)
            x_vis_enc = self.mid_project(m)
        elif self.direction == 'none':
            if isinstance(x, list) or isinstance(x, tuple):
                x_vis_enc = x[-1]
            else:
                x_vis_enc = x

        y_2d = y.squeeze().unsqueeze(-1).unsqueeze(-1)
        x_multi_modal = self.activate(self.linear(x_vis_enc)) * self.activate(y_2d)

        return x_multi_modal
    

@FUSIONS.register_module()
class SimpleFusionv3(nn.Module):
    def __init__(self,
                 vis_chs=(256, 512, 1024),
                 direction='bottom_up',
                 fusion_dim=1024,
                 ):
        super(SimpleFusionv3, self).__init__()
        self.fp16_enabled = False
        assert direction in ['to_mid', 'bottom_up', 'none']
        self.direction = direction

        if direction == 'bottom_up':
            assert len(vis_chs) == 3
            ch = sum(vis_chs[:2])
            self.down_mid2top = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((ch, ), (ch, ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.top_project = nn.Sequential(
                *darknet_conv((ch+vis_chs[-1], ch+vis_chs[-1], ), (ch+vis_chs[-1], vis_chs[-1], ), (3, 1, ), (1, 1, )))
        elif direction == 'to_mid':
            assert len(vis_chs) == 3
            ch = sum(vis_chs)
            self.up_top2mid = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                *darknet_conv((vis_chs[-1], ), (vis_chs[-1], ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.mid_project = nn.Sequential(
                *darknet_conv((ch, ch, ), (ch, vis_chs[-1], ), (3, 1, ), (3, 1, )))
        elif direction == 'none':
            pass
            

        self.activate = nn.Tanh()
        self.linear_vision = nn.Sequential(*darknet_conv((vis_chs[-1], ), (fusion_dim, ), (1, ), (1, )))
        self.linear_text = nn.Sequential(*darknet_conv((1024, ), (fusion_dim, ), (1, ), (1, )))
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x, y):
        """Args
            x (list[tensor]): multi-scale feature maps from visual encoders,
                e.g., for darknet, it's [batch_size, 256, 52, 52], [batch_size, 512, 26, 26],
                [batch_size, 1024, 13, 13] in sequential order.

            y (tensor): [batch_size, 1, C_l].

            y_word (tensor): [batch_size, max_token, C_l].

            y_mask (tensor): [batch_size, max_token], dtype=torch.bool, 
                True means ignored position.

        Returns:
            x_multi_modal (tensor): [batch_size, 512, 13, 13]
        """
        if self.direction == 'bottom_up':
            l, m, s = x
            m = torch.cat([self.down_bot2mid(l), m], 1)
            s = torch.cat([self.down_mid2top(m), s], 1)
            x_vis_enc = self.top_project(s)
        elif self.direction == 'to_mid':
            l, m, s = x
            s = self.up_top2mid(s)
            l = self.down_bot2mid(l)
            m = torch.cat([s, m, l], 1)
            x_vis_enc = self.mid_project(m)
        elif self.direction == 'none':
            if isinstance(x, list):
                x_vis_enc = x[-1]
            else:
                x_vis_enc = x

        y_2d = y.squeeze().unsqueeze(-1).unsqueeze(-1)
        # x_multi_modal = self.activate(self.linear(x_vis_enc))
        # x_multi_modal = self.linear(x_vis_enc) * y_2d
        x_multi_modal = self.activate(self.linear_vision(x_vis_enc*self.scale)) * self.activate(self.linear_text(y_2d))

        return x_multi_modal
    
    
    
@FUSIONS.register_module()
class SimpleFusionv4(nn.Module):
    def __init__(self,
                 vis_chs=(256, 512, 1024),
                 direction='bottom_up',
                 fusion_dim=1024,
                 ):
        super(SimpleFusionv4, self).__init__()
        self.fp16_enabled = False
        assert direction in ['to_mid', 'bottom_up', 'none']
        self.direction = direction

        if direction == 'bottom_up':
            assert len(vis_chs) == 3
            ch = sum(vis_chs[:2])
            self.down_mid2top = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((ch, ), (ch, ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.top_project = nn.Sequential(
                *darknet_conv((ch+vis_chs[-1], ch+vis_chs[-1], ), (ch+vis_chs[-1], vis_chs[-1], ), (3, 1, ), (1, 1, )))
        elif direction == 'to_mid':
            assert len(vis_chs) == 3
            ch = sum(vis_chs)
            self.up_top2mid = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                *darknet_conv((vis_chs[-1], ), (vis_chs[-1], ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.mid_project = nn.Sequential(
                *darknet_conv((ch, ch, ), (ch, vis_chs[-1], ), (3, 1, ), (3, 1, )))
        elif direction == 'none':
            pass

        self.activate = nn.Tanh()
        self.linear = nn.Conv2d(vis_chs[-1],fusion_dim,1,1)

    def forward(self, x, y):
        """Args
            x (list[tensor]): multi-scale feature maps from visual encoders,
                e.g., for darknet, it's [batch_size, 256, 52, 52], [batch_size, 512, 26, 26],
                [batch_size, 1024, 13, 13] in sequential order.

            y (tensor): [batch_size, 1, C_l].

            y_word (tensor): [batch_size, max_token, C_l].

            y_mask (tensor): [batch_size, max_token], dtype=torch.bool, 
                True means ignored position.

        Returns:
            x_multi_modal (tensor): [batch_size, 512, 13, 13]
        """
        if self.direction == 'bottom_up':
            l, m, s = x
            m = torch.cat([self.down_bot2mid(l), m], 1)
            s = torch.cat([self.down_mid2top(m), s], 1)
            x_vis_enc = self.top_project(s)
        elif self.direction == 'to_mid':
            l, m, s = x
            s = self.up_top2mid(s)
            l = self.down_bot2mid(l)
            m = torch.cat([s, m, l], 1)
            x_vis_enc = self.mid_project(m)
        elif self.direction == 'none':
            if isinstance(x, list) or isinstance(x, tuple):
                x_vis_enc = x[-1]
            else:
                x_vis_enc = x

        # y_2d = y.squeeze().unsqueeze(-1).unsqueeze(-1)
        # x_multi_modal = self.activate(self.linear(x_vis_enc)) * self.activate(y_2d)
        # x_multi_modal = self.linear(x_vis_enc)
        return x_vis_enc