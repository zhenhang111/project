import numpy as np
from elastica.external_forces import NoForces
from scipy.interpolate import make_interp_spline


class MuscleTorquesWithVaryingBetaSplines(NoForces):
    """

    This class compute the muscle torques using Beta spline.
    Points of beta spline can be changed through out the simulation, and
    every time it changes a new spline generated. Control algorithm has to
    select the spline control points. Location of control points on the arm
    is fixed and they are equidistant points.

    Attributes
    ----------
    direction : str
        Depending on the user input direction, computed torques are applied in the direction of d1, d2, or d3.
    points_array : numpy.ndarray or callable object
        This variable is a reference to points_func_array variable which can be a numpy.ndarray or callable object.
    base_length : float
        Initial length of the arm.
    muscle_torque_scale : float
        Scaling factor for beta spline muscle torques. Beta spline is non-dimensional and muscle_torque_scale scales it.
    torque_profile_recorder : defaultdict(list)
        This is a dictionary to store time-history of muscle torques and beta-spline.
    step_skip : int
        Determines the data collection step.
    counter : int
        Used to determine the current call step of this object.
    number_of_control_points : int
        Number of control points used in beta spline. Note that these are the control points in the middle and there
        are two more control points at the start and end of the rod, which are 0.
    points_cached : numpy.ndarray
        2D (2, number_of_control_points+2) array containing data with 'float' type.
        This array stores the location of control points in first row and in the second row it stores the values of
        control points selected at previous step. If control points are changed, points_cached updated.
    max_rate_of_change_of_control_points : float
        This limits the maximum change that can happen for control points in between two calls of this object.
    my_spline : object
        Stores the beta spline object generated by control points.
    """

    def __init__(
        self,
        base_length,
        number_of_control_points,
        points_func_array,
        muscle_torque_scale,
        direction,
        step_skip,
        max_rate_of_change_of_activation=0.01,
        **kwargs,
    ):
        """

        Parameters
        ----------
        base_length : float
            Initial length of the arm.
        number_of_control_points : int
            Number of control points used in beta spline. Note that these are the control points in the middle and there
            are two more control points at the start and end of the rod, which are 0.
        points_func_array  : numpy.ndarray
            2D (2, number_of_control_points+2) array containing data with 'float' type.
            This array stores the location of control points in first row and in the second row it stores the values of
            control points selected at previous step. If control points are changed, points_cached updated.
        muscle_torque_scale : float
            Scaling factor for beta spline muscle torques. Beta spline is non-dimensional and muscle_torque_scale
            scales it.
        direction  : str
            Depending on the user input direction, computed torques are applied in the "normal", "binormal", "tangent".
        step_skip  : int
            Determines the data collection step.
        max_rate_of_change_of_control_points : float
            This limits the maximum change that can happen for control points in between two calls of this object.
        **kwargs
            Arbitrary keyword arguments.
        """
        super(MuscleTorquesWithVaryingBetaSplines, self).__init__()

        if direction == str("normal"):
            self.direction = int(0)
        elif direction == str("binormal"):
            self.direction = int(1)
        elif direction == str("tangent"):
            self.direction = int(2)
        else:
            raise NameError(
                "Please type normal, binormal or tangent as muscle torque direction. Input should be string."
            )

        self.points_array = (
            points_func_array
            if hasattr(points_func_array, "__call__")
            else lambda time_v: points_func_array
        )

        self.base_length = base_length
        self.muscle_torque_scale = muscle_torque_scale

        self.torque_profile_recorder = kwargs.get("torque_profile_recorder", None)
        self.step_skip = step_skip
        self.counter = 0  # for recording data from the muscles
        self.number_of_control_points = number_of_control_points
        self.points_cached = np.zeros(
            (2, self.number_of_control_points + 2)
        )  # This caches the control points. Note that first and last control points are zero.
        self.points_cached[0, :] = np.linspace(
            0, self.base_length, self.number_of_control_points + 2
        )  # position of control points along the rod.

        # Max rate of change of activation determines, maximum change in activation
        # signal in one time-step.
        self.max_rate_of_change_of_activation = max_rate_of_change_of_activation

        # Purpose of this flag is to just generate spline even the control points are zero
        # so that code wont crash.
        self.initial_call_flag = 0

    def apply_torques(self, system, time: np.float = 0.0):

        # Check if RL algorithm changed the points we fit the spline at this time step
        # if points_array changed create a new spline. Using this approach we don't create a
        # spline every time step.
        # Make sure that first and last point y values are zero. Because we cannot generate a
        # torque at first and last nodes.
        if (
            not np.array_equal(self.points_cached[1, 1:-1], self.points_array(time))
            or self.initial_call_flag == 0
        ):
            self.initial_call_flag = 1

            # Apply filter to the activation signal, to prevent drastic changes in activation signal.
            self.filter_activation(
                self.points_cached[1, 1:-1],
                np.array(self.points_array(time)),
                self.max_rate_of_change_of_activation,
            )

            # self.points_cached[1, 1:-1] = self.points_array(time)
            self.my_spline = make_interp_spline(
                self.points_cached[0], self.points_cached[1]
            )
            # Compute the muscle torque magnitude from the beta spline.
            self.torque_magnitude_cache = self.muscle_torque_scale * self.my_spline(
                np.cumsum(system.lengths)
            )

        system.external_torques[self.direction, :] += self.torque_magnitude_cache[:]

        if self.counter % self.step_skip == 0:
            if self.torque_profile_recorder is not None:
                self.torque_profile_recorder["time"].append(time)

                self.torque_profile_recorder["torque_mag"].append(
                    self.torque_magnitude_cache
                )
                self.torque_profile_recorder["torque"].append(
                    system.external_torques.copy()
                )
                self.torque_profile_recorder["element_position"].append(
                    np.cumsum(system.lengths)
                )

        self.counter += 1

    @staticmethod
    def filter_activation(signal, input_signal, max_signal_rate_of_change):
        """
        Filters the input signal. If change in new signal (input signal) greater than
        previous signal (signal) then, increase for signal is max_signal_rate_of_change amount.

        Parameters
        ----------
        signal : numpy.ndarray
            1D (number_of_control_points,) array containing data with 'float' type.
        input_signal : numpy.ndarray
            1D (number_of_control_points,) array containing data with 'float' type.
        max_signal_rate_of_change : float
            This limits the maximum change that can happen between signal and input signal.

        Returns
        -------

        """
        signal_difference = input_signal - signal
        signal += np.sign(signal_difference) * np.minimum(
            max_signal_rate_of_change, np.abs(signal_difference)
        )
