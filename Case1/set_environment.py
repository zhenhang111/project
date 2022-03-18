__doc__ = """This file is for setting an environment for  arm following a randomly moving target. 
Actuation torques acting on arm can generate torques in normal, binormal and tangent 
direction. Environment set in this file is interfaced with stable-baselines and OpenAI Gym. It is shown that this
environment works with PPO, TD3, DDPG, TRPO and SAC."""
'''设置软体臂随着移动目标移动的环境，作用在软体臂上的驱动力矩可以产生法线方向、副法线方向和切线方向的力矩。
此文件与stable-baselines、 gym 设置有接口
'''

import gym
from gym import spaces
import copy

from post_processing import plot_video_with_sphere, plot_video_with_sphere_2D
#后处理文件  用来绘制视频

from MuscleTorquesWithBspline.BsplineMuscleTorques import (
    MuscleTorquesWithVaryingBetaSplines,
)
#导入肌肉力的文件

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *

# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass
#上句定义基本的模拟器类

class Environment(gym.Env):
    """
    遵循OpenAI Gym接口的自定 义环境。此环境下，生成一个臂(cosserat杆)和靶(刚体球)。靶球是不断移动的，其移动由用户自己定义
    控制器必须选择控制点所在，存储在动作中并且输入到步进类方法。控制点范围为【-1，1】，用来生成β样条。β样条通过沿着软体臂计算得到的
    扭矩因子（β或者α）进行缩放。肌肉力用来弯曲或者旋转手臂去跟踪移动的目标。


    Custom environment that follows OpenAI Gym interface. This environment, generates an
    arm (Cosserat rod) and target (rigid sphere). Target is moving throughout the simulation in a space, defined
    by the user. Controller has to select control points (stored in action) and input to step class method.
    Control points have to be in between [-1,1] and are used to generate a beta spline. This beta spline is scaled
    by the torque scaling factor (alpha or beta) and muscle torques acting along arm computed. Muscle torques bend
    or twist the arm and tracks the moving target.

    Attributes                          参数
    ----------
    dim : float                         维度：float类型
        Dimension of the problem.           问题的维度
        If dim=2.0 2D problem only muscle torques in normal direction is activated.
        如果维度是2，则表明只有法线方向的力
        If dim=2.5 or 3.0 3D problem muscle torques in normal and binormal direction are activated.
        维数为2.5或者3 激活法线和副法线方向的力
        If dim=3.5 3D problem muscle torques in normal, binormal and tangent direction are activated.
        维数为3.5 激活主副法线、切线方向的力
    n_elem : int
        Cosserat rod number of elements.        分段数
    final_time : float
        Final simulation time.                  最终仿真时间
    time_step : float
        Simulation time-step.                   仿真步数
    number_of_control_points : int
        Number of control points for beta-spline that generate muscle torques.
        肌肉力的控制点数

    alpha : float
        Muscle torque scaling factor for normal/binormal directions.
        法线/副法线方向的肌肉扭矩比例因子

    beta : float
        Muscle torque scaling factor for tangent directions (generates twist).
        切线方向的肌肉扭矩缩放因子(产生扭转)。

    target_position :  numpy.ndarray
        1D (3,) array containing data with 'float' type.
        目标位置
        Initial target position, If mode is 2 or 4 target randomly placed.
        初始目标位置，如果是模式2或者4，则目标位置随机

    num_steps_per_update : int
        Number of Elastica simulation steps, before updating the actions by control algorithm.
        仿真步骤数，在控制算法更新动作之前，就是每次计算下一次的状态所耗费的步骤数
    action : numpy.ndarray
        1D (n_torque_directions * number_of_control_points,) array containing data with 'float' type.
        一维数组，float类型。
        Action returns control points selected by control algorithm to the Elastica simulation.
        动作将控制算法选择的控制点返回给Elastica仿真。
        n_torque_directions is number of torque directions, this is controlled by the dim.
        n_torque_directions是力矩方向的个数，这是由dim控制的。

    action_space : spaces.Box
        1D (n_torque_direction * number_of_control_poinst,) array containing data with 'float' type in range [-1., 1.].
        gym里的action一种存储方式，为一维数组，范围为[-1,1].
    obs_state_points : int
        Number of arm (Cosserat rod) points used for state information.
        软体臂的状态点，用于展示软体臂的状态信息
    number_of_points_on_cylinder : int
        Number of cylinder points used for state information.
        柱状物的状态点（障碍物）
    observation_space : spaces.Box
        1D ( total_number_of_states,) array containing data with 'float' type.
        State information of the systems are stored in this variable.
        观测空间：系统的状态信息存储在这个变量中。
    mode : int                          软体臂的工作模式：四种
        There are 4 modes available.
        mode=1 fixed target position to be reached (default)
        mode=2 randomly placed fixed target position to be reached. Target position changes every reset call.
        mode=3 moving target on fixed trajectory.
        mode=4 randomly moving target.
    COLLECT_DATA_FOR_POSTPROCESSING : boolean
        If true data from simulation is collected for post-processing. If false post-processing making videos and storing data is not done.
        是否从模拟中采集数据进行后处理。如果选择为false，制作视频和存储数据是不做的。
    E : float
        Young's modulus of the arm (Cosserat rod).    杨氏模量
    NU : float
        Dissipation constant of the arm (Cosserat rod).         μ  耗散常数
    COLLECT_CONTROL_POINTS_DATA : boolean
        If true actions or selected control points by the controller are stored throughout the simulation.
        如果为真，控制器的动作或选择的控制点在整个模拟过程中被存储。
    total_learning_steps : int
        Total number of steps, controller is called. Also represents how many times actions changed throughout the simulation.
        总学习步数，也表示在整个模拟过程中动作改变的次数。
    control_point_history_array : numpy.ndarray
         2D (total_learning_steps, number_of_control_points) array containing data with 'float' type.
         Stores the actions or control points selected by the controller.
         存储控制器选择的操作或控制点。
    shearable_rod : object
        shearable_rod or arm is Cosserat Rod object.  杆对象
    sphere : object
        Target sphere is rigid Sphere object.       目标点
    spline_points_func_array_normal_dir : list
        Contains the control points for generating spline muscle torques in normal direction.
        包含法向方向上产生肌肉力矩的控制点。
    torque_profile_list_for_muscle_in_normal_dir : defaultdict(list)
        Records, muscle torques and control points in normal direction throughout the simulation.
        记录法线上的肌肉力
    spline_points_func_array_binormal_dir : list
        Contains the control points for generating spline muscle torques in binormal direction.
        副法线上产生肌肉样条的控制点
    torque_profile_list_for_muscle_in_binormal_dir : defaultdict(list)
        Records, muscle torques and control points in binormal direction throughout the simulation.
        副法线上肌肉及控制点
    spline_points_func_array_tangent_dir : list
        Contains the control points for generating spline muscle torques in tangent direction.
        切线方向
    torque_profile_list_for_muscle_in_tangent_dir : defaultdict(list)
        Records, muscle torques and control points in tangent direction throughout the simulation.
        切线方向肌肉力
    post_processing_dict_rod : defaultdict(list)
        Contains the data collected by rod callback class. It stores the time-history data of rod and only initialized
        if COLLECT_DATA_FOR_POSTPROCESSING=True.
        当flag为真时，进行后处理，包含由棒回调类收集的数据。
    post_processing_dict_sphere : defaultdict(list)
        Contains the data collected by target sphere callback class. It stores the time-history data of rod and only
        initialized if COLLECT_DATA_FOR_POSTPROCESSING=True.
        目标点的回调，
    step_skip : int
        Determines the data collection step for callback functions. Callback functions collect data every step_skip.
        确定回调函数的数据收集步骤。回调函数在每次step_skip时收集数据。
    """

    # Required for OpenAI Gym interface
    metadata = {"render.modes": ["human"]}      #这里时一个键值对，渲染到当前显示器或终端，不返回任何东西。render一共有三种类型，rgb_array human ansi

    """
    
    以下是四种状态
    FOUR modes: (specified by mode)
    1. fixed target position to be reached (default: need target_position parameter)
    2. random fixed target position to be reached
    3. fixed trajectory to be followed
    4. random trajectory to be followed (no need to worry in current phase)
    """

    def __init__(
        self,
        final_time,
        num_steps_per_update,
        number_of_control_points,
        alpha,
        beta,
        target_position,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=2.5e-4,
        n_elem=40,
        mode=1,
        dim=3.5,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        final_time : float
            Final simulation time.
        n_elem : int
            Arm (Cosserat rod) number of elements.                  分段
        num_steps_per_update : int
            Number of Elastica simulation steps, before updating the actions by control algorithm.
        number_of_control_points : int
            Number of control points for beta-spline that generate muscle torques.
        alpha : float
            Muscle torque scaling factor for normal/binormal directions.
        beta : float
            Muscle torque scaling factor for tangent directions (generates twist).
        target_position :  numpy.ndarray
            1D (3,) array containing data with 'float' type.
            Initial target position, If mode is 2 or 4 target randomly placed.
        COLLECT_DATA_FOR_POSTPROCESSING : boolean
            If true data from simulation is collected for post-processing. If false post-processing making videos
            and storing data is not done.
        sim_dt : float
            Simulation time-step
        mode : int
            There are 4 modes available.
            mode=1 fixed target position to be reached (default)
            mode=2 randomly placed fixed target position to be reached. Target position changes every reset call.
            mode=3 moving target on fixed trajectory.
            mode=4 randomly moving target.
        num_obstacles : int
            Number of rigid cylinder obstacles.
        COLLECT_CONTROL_POINTS_DATA : boolean
            If true actions or selected control points by the controller are stored throughout the simulation.
        *args
            Variables length arguments. Currently, *args are not used.
        **kwargs
            Arbitrary keyword arguments.
            * E : float
                Young's modulus of the arm (Cosserat rod). Default 1e7Pa            杨氏模量 默认为10的七次方（生物组织的软度）
            * NU : float
                Dissipation constant of the arm (Cosserat rod). Default 10.         耗散常数
            * target_v : float
                Target velocity for moving taget, if mode=3,4 it is used.           目标运动速度
            * boundary : numpy.ndarray
                1D (6,) array containing data with 'float' type.                    边界条件
                boundary used if mode=2,4. It determines the rectangular space, that target can move and  minimum
                and maximum of this space are given for x, y, and z coordinates. (xmin, xmax, ymin, ymax, zmin, zmax)

        """
        super(Environment, self).__init__()
        self.dim = dim
        # Integrator type                                   #整型变量
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters                             #仿真参数
        self.final_time = final_time
        self.h_time_step = sim_dt               # this is a stable time step     固定的时间步长
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        print("Total steps", self.total_steps)                          #打印出总的仿真步数

        # Video speed                                                   视频参数
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Number of control points                                      控制点个数
        self.number_of_control_points = number_of_control_points

        # Actuation torque scaling factor in normal/binormal direction          驱动力矩比例系数（主副法线方向）α
        self.alpha = alpha

        # Actuation torque scaling factor in tangent direction                  同上 （切线方向）
        self.beta = beta

        # target position                                                       目标点位置
        self.target_position = target_position

        # learning step define through num_steps_per_update                     学习的次数（更新的次数=总步数除以间隔）
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        if self.dim == 2.0:
            # normal direction activation (2D)                                  法线方向激活
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(self.number_of_control_points)
        if self.dim == 3.0 or self.dim == 2.5:
            # normal and/or binormal direction activation (3D)              法线或者副法线激活
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(2 * self.number_of_control_points)
        if self.dim == 3.5:
            # normal, binormal and/or tangent direction activation (3D)         三维，Box空间里含有十八个数
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(3 * self.number_of_control_points)

        self.obs_state_points = 10                                      #软体臂观测点个数
        num_points = int(n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(n_elem + 1)[0::num_points])         #

        # 8: 4 points for velocity and 4 points for orientation         4速度4方向
        # 11: 3 points for target position plus 8 for velocity and orientation  3目标位置信息 4速度4方向
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 3 + 11,),
            dtype=np.float64,
        )

        # here we specify 4 tasks that can possibly used            #4项任务
        self.mode = mode

        if self.mode is 2:
            assert "boundary" in kwargs, "need to specify boundary in mode 2"       #模式二中需要指定边界
            self.boundary = kwargs["boundary"]

        if self.mode is 3:
            assert "target_v" in kwargs, "need to specify target_v in mode 3"       #模式三中需要指定target_v
            self.target_v = kwargs["target_v"]

        if self.mode is 4:
            assert (
                "boundary" and "target_v" in kwargs
            ), "need to specify boundary and target_v in mode 4"                    #模式4中指定以上两种
            self.boundary = kwargs["boundary"]
            self.target_v = kwargs["target_v"]

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.                                 #这个参数如果定义了，则会列出一个单子，供回调函数调用生成清单
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

        self.time_tracker = np.float64(0.0)

        self.acti_diff_coef = kwargs.get("acti_diff_coef", 9e-1)

        self.acti_coef = kwargs.get("acti_coef", 1e-1)

        self.max_rate_of_change_of_activation = kwargs.get(                     #不让波动太大
            "max_rate_of_change_of_activation", np.infty
        )

        self.E = kwargs.get("E", 1e7)

        self.NU = kwargs.get("NU", 10)

        self.n_elem = n_elem

    def reset(self):
        """

        This class method, resets and creates the simulation environment. First,
        Elastica rod (or arm) is initialized and boundary conditions acting on the rod defined.
        Second, target and if there are obstacles are initialized and append to the
        simulation. Finally, call back functions are set for Elastica rods and rigid bodies.

        该类方法重置并创建模拟环境。首先，初始化弹性杆（或臂），并定义作用在杆上的边界条件。
        第二，初始化目标和障碍物，并将其附加到模拟中。
        最后，为弹性杆和刚体设置回调函数。

        Returns
        -------

        """
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = self.n_elem
        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction: pointing upwards        竖直向上
        normal = np.array([0.0, 0.0, 1.0])                                              #法线方向
        binormal = np.cross(direction, normal)                                          #副法线方向（前两者叉乘）

        density = 1000
        nu = self.NU  # dissipation coefficient                         #耗散常数
        E = self.E  # Young's Modulus
        poisson_ratio = 0.5                                             #泊松比

        # Set the arm properties after defining rods                    机械臂参数
        base_length = 1.0  # rod base length                            基本长度
        radius_tip = 0.05  # radius of the arm at the tip                   顶端半径
        radius_base = 0.05  # radius of the arm at the base                 基底半径

        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)         #半径均匀变化

        # Arm is shearable Cosserat rod                                 #具有剪切应变的弹性杆
        self.shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius=radius_along_rod,
            density=density,
            nu=nu,
            youngs_modulus=E,
            poisson_ratio=poisson_ratio,
        )

        # Now rod is ready for simulation, append rod to simulation         基本参数至此已设定好，将其添加到模拟器中
        self.simulator.append(self.shearable_rod)                            #加进去
        # self.mode = 4
        if self.mode != 2:
            # fixed target position to reach                        #固定目标
            target_position = self.target_position

        if self.mode == 2 or self.mode == 4:
            # random target position to reach with boundary             随机目标位置但是有边界
            t_x = np.random.uniform(self.boundary[0], self.boundary[1])         #boundary是6元数，前两个为x轴，后两个z轴，中间为y轴
            t_y = np.random.uniform(self.boundary[2], self.boundary[3])
            if self.dim == 2.0 or self.dim == 2.5:
                t_z = np.random.uniform(self.boundary[4], self.boundary[5]) * 0
            elif self.dim == 3.0 or self.dim == 3.5:
                t_z = np.random.uniform(self.boundary[4], self.boundary[5])

            print("Target position:", t_x, t_y, t_z)
            target_position = np.array([t_x, t_y, t_z])                     #确定出位置信息 target_position

        # initialize sphere
        self.sphere = Sphere(
            center=target_position,  # initialize target position of the ball       球形目标点
            base_radius=0.05,
            density=1000,                               # 为啥要定密度？？
        )

        if self.mode == 3:
            self.dir_indicator = 1                                  #方向指示
            self.sphere_initial_velocity = self.target_v                    #Mode3情况下设置目标运动速度
            self.sphere.velocity_collection[..., 0] = [
                self.sphere_initial_velocity,
                0.0,
                0.0,
            ]

        if self.mode == 4:

            self.trajectory_iteration = 0  # for changing directions            #轨迹迭代，改变方向
            self.rand_direction_1 = np.pi * np.random.uniform(0, 2)             #方向1、2
            if self.dim == 2.0 or self.dim == 2.5:
                self.rand_direction_2 = np.pi / 2.0
            elif self.dim == 3.0 or self.dim == 3.5:
                self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

            self.v_x = (
                self.target_v
                * np.cos(self.rand_direction_1)
                * np.sin(self.rand_direction_2)
            )
            self.v_y = (                                                #各方向上的分量
                self.target_v
                * np.sin(self.rand_direction_1)
                * np.sin(self.rand_direction_2)
            )
            self.v_z = self.target_v * np.cos(self.rand_direction_2)

            self.sphere.velocity_collection[..., 0] = [
                self.v_x,
                self.v_y,
                self.v_z,
            ]                                                           #目标球的速度，一个数组表示 xyz
            self.boundaries = np.array(self.boundary)                   #边界信息

        # Set rod and sphere directors to each other.
        self.sphere.director_collection[
            ..., 0
        ] = self.shearable_rod.director_collection[..., 0]
        self.simulator.append(self.sphere)

        class WallBoundaryForSphere(FreeRod):
            """

            This class generates a bounded space that sphere can move inside. If sphere
            hits one of the boundaries (walls) of this space, it is reflected in opposite direction
            with the same velocity magnitude.

            这个类生成一个球体可以在其中移动的有界空间。如果球碰到了这个空间的边界(墙),它以相反的方向反射同样的速度大小。
            """

            def __init__(self, boundaries):
                self.x_boundary_low = boundaries[0]
                self.x_boundary_high = boundaries[1]
                self.y_boundary_low = boundaries[2]
                self.y_boundary_high = boundaries[3]
                self.z_boundary_low = boundaries[4]
                self.z_boundary_high = boundaries[5]

            def constrain_values(self, sphere, time):
                pos_x = sphere.position_collection[0]
                pos_y = sphere.position_collection[1]
                pos_z = sphere.position_collection[2]

                radius = sphere.radius

                vx = sphere.velocity_collection[0]
                vy = sphere.velocity_collection[1]
                vz = sphere.velocity_collection[2]

                if (pos_x - radius) < self.x_boundary_low:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_x + radius) > self.x_boundary_high:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_y - radius) < self.y_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_y + radius) > self.y_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_z - radius) < self.z_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

                if (pos_z + radius) > self.z_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

            def constrain_rates(self, sphere, time):
                pass

        if self.mode == 4:                                  #mode4情况下设置边界
            self.simulator.constrain(self.sphere).using(
                WallBoundaryForSphere, boundaries=self.boundaries
            )

        # Add boundary constraints as fixing one end            固定一段
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        # Add muscle torques acting on the arm for actuation        增加作用在手臂上的肌肉力矩来驱动
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected by RL to
        # generate torques along the arm.
        #
        # MuscleTorquesWithVaryingBetaSplines    使用RL选择的控制点来沿手臂产生力矩。
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        # Apply torques                                                         将法线方向上的力加入
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )
        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        # Apply torques                                                 #副法线力加入
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir = []
        # Apply torques                                                         #切线方向力加入
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_twist_dir,
        )

        # Call back function to collect arm data from simulation                回调函数（手臂）
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            """
            Call back function for Elastica rod
            """

            def __init__(
                self, step_skip: int, callback_params: dict,
            ):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
        #信息包含：时间、当前步数、位置信息、半径、质心
                    return

        # Call back function to collect target sphere data from simulation    回调函数（目标球）
        class RigidSphereCallBack(CallBackBaseClass):
            """
            Call back function for target sphere
            """

            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing           回调函数收集（调用前边两个函数）
            self.post_processing_dict_rod = defaultdict(list)
            # list which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,                  #手臂
            )

            self.post_processing_dict_sphere = defaultdict(list)
            # list which collected data will be append
            # set the diagnostics for cyclinder and collect data                #目标球
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
            )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()                                               #系统最终确定

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(           #这里是关于时间步进的，在step中调用
            self.StatefulStepper, self.simulator
        )

        # set state
        state = self.get_state()                               #状态信息

        # reset on_goal
        self.on_goal = 0
        # reset current_step
        self.current_step = 0                               #参数的重置
        # reset time_tracker
        self.time_tracker = np.float64(0.0)
        # reset previous_action
        self.previous_action = None

        # After resetting the environment return state information      重置环境后返回状态信息
        return state

    def sampleAction(self):
        """
        Sample usable random actions are returned.

        :返回一个随机动作

        Returns
        -------
        numpy.ndarray
            1D (3 * number_of_control_points,) array containing data with 'float' type, in range [-1, 1].
        """
        random_action = (np.random.rand(1 * self.number_of_control_points) - 0.5) * 2
        return random_action

    def get_state(self):
        """
        Returns current state of the system to the controller.

        用于返回当前状态

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """

        rod_state = self.shearable_rod.position_collection
        r_s_a = rod_state[0]  # x_info
        r_s_b = rod_state[1]  # y_info
        r_s_c = rod_state[2]  # z_info

        num_points = int(self.n_elem / self.obs_state_points)
        ## get full 3D state information
        rod_compact_state = np.concatenate(
            (
                r_s_a[0 : len(r_s_a) + 1 : num_points],
                r_s_b[0 : len(r_s_b) + 1 : num_points],
                r_s_c[0 : len(r_s_b) + 1 : num_points],
            )
        )

        rod_compact_velocity = self.shearable_rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([np.linalg.norm(rod_compact_velocity)])        #范数
        rod_compact_velocity_dir = np.where(
            rod_compact_velocity_norm != 0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.sphere.position_collection.flatten()  # 2
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array(
            [np.linalg.norm(sphere_compact_velocity)]
        )
        sphere_compact_velocity_dir = np.where(
            sphere_compact_velocity_norm != 0,
            sphere_compact_velocity / sphere_compact_velocity_norm,
            0.0,
        )

        state = np.concatenate(
            (
                # rod information
                rod_compact_state,
                rod_compact_velocity_norm,                #包括软体臂的位置状态信息（33）、范数、速度和目标点的位置、速度信息
                rod_compact_velocity_dir,
                # target information
                sphere_compact_state,
                sphere_compact_velocity_norm,
                sphere_compact_velocity_dir,
            )
        )

        return state

    def step(self, action):
        """
        This method integrates the simulation number of steps given in num_steps_per_update, using the actions
        selected by the controller and returns state information, reward, and done boolean.

        主要的函数，学习过程中的迭代、下一状态、奖励、判断是否完成等均在此函数中完成。
        Parameters
        ----------
        action :  numpy.ndarray
            1D (n_torque_directions * number_of_control_points,) array containing data with 'float' type.
            Action returns control points selected by control algorithm to the Elastica simulation. n_torque_directions
            is number of torque directions, this is controlled by the dim.

            动作将控制算法选择的控制点返回到Elastica模拟。
n_torque_directions：力矩方向的个数，由dim控制

        Returns             此函数返回state reward done 分别为状态，奖励，是否完成任务的标志
        -------
        state : numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        reward : float
            Reward after the integration.
        done: boolean
            Stops, simulation or training if done is true. This means, simulation reached final time or NaN is
            detected in the simulation.

        """

        # action contains the control points for actuation torques in different directions in range [-1, 1]
        #动作包含[-1, 1]范围内不同方向驱动扭矩的控制点
        self.action = action

        # set binormal activations to 0 if solving 2D case      2d情况下副法线不激活
        if self.dim == 2.0:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 2.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = action[
                self.number_of_control_points :
            ]
        # apply binormal activations if solving 3D case             3d 副法线激活
        elif self.dim == 3.0:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = action[
                self.number_of_control_points :
            ]
            self.spline_points_func_array_twist_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 3.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = action[
                self.number_of_control_points : 2 * self.number_of_control_points
            ]
            self.spline_points_func_array_twist_dir[:] = action[
                2 * self.number_of_control_points :
            ]

        # Do multiple time step of simulation for <one learning step>   对<一个学习步>做多个时间步模拟
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        if self.mode == 3:                                               #模式3情况下运行
            ##### (+1, 0, 0) -> (0, -1, 0) -> (-1, 0, 0) -> (0, +1, 0) -> (+1, 0, 0) #####
            if (
                self.current_step
                % (1.0 / (self.h_time_step * self.num_steps_per_update))
                == 0
            ):
                if self.dir_indicator == 1:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        -self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 2
                elif self.dir_indicator == 2:
                    self.sphere.velocity_collection[..., 0] = [
                        -self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 3
                elif self.dir_indicator == 3:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        +self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 4
                elif self.dir_indicator == 4:
                    self.sphere.velocity_collection[..., 0] = [
                        +self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 1
                else:
                    print("ERROR")

        if self.mode == 4:
            self.trajectory_iteration += 1
            if self.trajectory_iteration == 500:
                # print('changing direction')
                self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
                if self.dim == 2.0 or self.dim == 2.5:
                    self.rand_direction_2 = np.pi / 2.0
                elif self.dim == 3.0 or self.dim == 3.5:
                    self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

                self.v_x = (
                    self.target_v
                    * np.cos(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
                )
                self.v_y = (
                    self.target_v
                    * np.sin(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
                )
                self.v_z = self.target_v * np.cos(self.rand_direction_2)

                self.sphere.velocity_collection[..., 0] = [
                    self.v_x,
                    self.v_y,
                    self.v_z,
                ]
                self.trajectory_iteration = 0

        self.current_step += 1                              #步数+1

        # observe current state: current as sensed signal
        state = self.get_state()                        #获取当前位置信息

        # print(self.sphere.position_collection[..., 0])
        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )                                                   #和目标点的距离
        # print("和目标点的距离:")
        # print(dist)
        """ Reward Engineering """
        reward_dist = -np.square(dist).sum()            #总奖励

        reward = 1.0 * reward_dist                      #为啥还要乘以1  ？？？？
        """ Done is a boolean to reset the environment before episode is completed """
        done = False                                    #标志位

        # Position of the rod cannot be NaN, it is not valid, stop the simulation       检查杆的位置是不是不存在
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print(" Nan detected, exiting simulation now")
            print(" 期望外位置信息出现, 正在退出模拟")
            self.shearable_rod.position_collection = np.zeros(
                self.shearable_rod.position_collection.shape
            )
            reward = -1000
            state = self.get_state()
            done = True

        if np.isclose(dist, 0.0, atol=0.05 * 2.0).all():        #比较dist和0的接近程度
            self.on_goal += self.time_step
            reward += 0.5
        # for this specific case, check on_goal parameter
        if np.isclose(dist, 0.0, atol=0.05).all():
            self.on_goal += self.time_step
            reward += 1.5

        else:
            self.on_goal = 0
        # print("on-goal:",self.on_goal)
        if self.current_step >= self.total_learning_steps:              #执行步骤大于设定学习步骤数，退出
            done = True
            if reward > 0:
                print(
                    " Reward greater than 0! Reward: %0.3f, Distance: %0.3f "
                    % (reward, dist)
                )
                print(
                    " 奖励大于0了！奖励：%0.3f, 距离：%0.3f "
                    % (reward, dist)
                )
            else:
                print(
                    " Finished simulation. Reward: %0.3f, Distance: %0.3f"
                    % (reward, dist)
                )
                print(
                    " 仿真结束...... 总奖励值: %0.3f, 距离: %0.3f"
                    % (reward, dist)
                )
        """ Done is a boolean to reset the environment before episode is completed """

        self.previous_action = action

        return state, reward, done, {"ctime": self.time_tracker}        #返回值：四个参数

    def render(self, mode="human"):
        """
        This method does nothing, it is here for interfacing with OpenAI Gym.
        没用，不重写 会报错
        Parameters
        ----------
        mode

        Returns
        -------

        """
        return

    def post_processing(self, filename_video, SAVE_DATA=False, **kwargs):
        """
        Make video 3D of arm movement in time, and store the arm, target, obstacles, and actuation
        data.

        生成手臂运动的3D视频，并存储手臂、目标、障碍物和驱动数据。
        Parameters
        ----------
        filename_video : str
            Names of the videos to be made for post-processing.
        SAVE_DATA : boolean
            If true collected data in simulation saved.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            plot_video_with_sphere_2D(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name="2d_" + filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            plot_video_with_sphere(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name="3d_" + filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            if SAVE_DATA == True:
                import os

                save_folder = os.path.join(os.getcwd(), "data")
                os.makedirs(save_folder, exist_ok=True)

                # Transform nodal to elemental positions
                position_rod = np.array(self.post_processing_dict_rod["position"])
                position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])

                np.savez(
                    os.path.join(save_folder, "arm_data.npz"),
                    position_rod=position_rod,
                    radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                    n_elems_rod=self.shearable_rod.n_elems,
                    position_sphere=np.array(
                        self.post_processing_dict_sphere["position"]
                    ),
                    radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
                )

                np.savez(
                    os.path.join(save_folder, "arm_activation.npz"),
                    torque_mag=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque_mag"]
                    ),
                    torque_muscle=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque"]
                    ),
                )

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )
