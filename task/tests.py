from hstest import *
from math import isclose


def is_float(num: str):
    try:
        float(num)
        return True
    except ValueError:
        return False


class Stage5Test(PlottingTest):
    def check_outputs_number(self, values_number: int, user_output: str):
        outputs = user_output.split()

        if not all(is_float(output) for output in outputs):
            raise WrongAnswer(f"Answer '{user_output}' contains non-numeric values.")

        if len(outputs) != values_number:
            raise WrongAnswer(f"Answer contains {len(outputs)} values, but {values_number} values are expected.")

    def check_num_values(self, values: list, user_values: list, message: str, rel_tol=1.0e-3):
        if not all(isclose(value, user_value, rel_tol=rel_tol) for value, user_value in zip(values, user_values)):
            raise WrongAnswer(message)

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        user_output = pr.start().strip()

        if len(user_output.strip()) == 0:
            raise WrongAnswer("Seems like your program does not show any output.")

        # check output format
        self.check_outputs_number(4, user_output)

        # check median separation for HCG 2 group
        hcg2_median_sep = [76.98939987756634]
        user_values = [float(value) for value in user_output.split()][:1]
        self.check_num_values(hcg2_median_sep, user_values, "The median separation in HCG 2 group is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        # check Shapiro tests p-values
        p_value = [0.6165452599525452]
        user_values = [float(value) for value in user_output.split()][1:2]
        self.check_num_values(p_value, user_values,
                              "The p-value of Shapiro-Wilk test for the projected median separation is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        p_value = [0.899420976638794]
        user_values = [float(value) for value in user_output.split()][2:3]
        self.check_num_values(p_value, user_values,
                              "The p-value of Shapiro-Wilk test for the mean surface brightness of the IGL is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        # check non-correlation test p-value
        p_values = [0.17506762692490038]
        user_values = [float(value) for value in user_output.split()][3:4]
        self.check_num_values(p_values, user_values, "The p-value of non-correlation test is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        return CheckResult.correct()


if __name__ == '__main__':
    Stage5Test().run_tests()
