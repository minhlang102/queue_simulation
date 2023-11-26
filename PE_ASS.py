import simpy
import numpy as np
import matplotlib.pyplot as plt
import random
from random import seed

LAM = 2000/60

MU1 = 1000/60
MU2 = 1000/60
MU3 = 1000/60
MU4 = 2000/60
MU5 = 300/60
MU6 = 500/60
MU7 = 500/60
MU8 = 500/60
MU9 = 500/60
MU10 = 500/60
MU11 = 500/60
MU12 = 500/60
MU13 = 100/60
NUMBER_CHECK_IN = 3
NUMBER_CHECK_INFO = 2
NUMBER_SERVER_OF_EACH_CHECK_INFO = 3
NUMBER_CHECK_SECURITY = 8
SIM_TIME = 60

p = [[0 for j in range(14)] for i in range(14)]

p[0][1] = 0.4
p[0][2] = 0.3
p[0][3] = 0.3

p[1][4] = 0.9
p[1][5] = 0.1

p[2][4] = 0.9
p[2][5] = 0.1

p[3][4] = 0.9
p[3][5] = 0.1

p[4][1] = 0.01
p[4][2] = 0.01
p[4][3] = 0.01
p[4][6] = 0.155
p[4][7] = 0.155
p[4][8]= 0.155
p[4][9]= 0.155
p[4][10] = 0.155
p[4][11] = 0.155
p[4][0] = 0.04

p[5][1] = 0.01
p[5][2] = 0.01
p[5][3] = 0.01
p[5][12] = 0.9
p[5][0] = 0.07

p[6][13] = 0.1
# p6_pass = 0.1

p[7][13] = 0.1
# p7_pass = 0.1

p[8][13] = 0.1
# p8_pass = 0.1

p[9][13 ]= 0.1
# p9_pass = 0.1

p[10][13] = 0.1
# p10_pass = 0.1

p[11][13] = 0.1
# p11_pass = 0.1

p[12][13] = 0.1
# p12_pass = 0.1

p[13][0] = 0.1
# p13_pass = 0.1

class CheckIn:
    def __init__(self, env, id, mu, number_of_server=1):
        self.id = id
        self.mu = mu
        self.resource = simpy.Resource(env, number_of_server)

    def doService(self, env, passenger_id):
        service_time = np.random.exponential(1.0 / self.mu)
        yield env.timeout(service_time)
        # print(f"Passenger {passenger_id} finish check in at {env.now}")


class CheckInfo:
    def __init__(self, env, id, mu, number_of_server=1):
        self.id = id
        self.mu = mu
        self.resource = simpy.Resource(env, number_of_server)

    def doService(self, env, passenger_id):
        service_time = np.random.exponential(1.0 / self.mu)
        yield env.timeout(service_time)
        # print(f"Passenger {passenger_id} finish check information at {env.now}")


class CheckSecurity:
    def __init__(self, env, id, mu, number_of_server=1):
        self.id = id
        self.mu = mu
        self.resource = simpy.Resource(env, number_of_server)

    def doService(self, env, passenger_id):
        service_time = np.random.exponential(1.0 / self.mu)
        yield env.timeout(service_time)
        # print(f"Passenger {passenger_id} finish check security {self.id} at {env.now}")


class Server:
    def __init__(self, check_in, check_info, check_security):
        self.check_in_server = check_in
        self.check_info_server = check_info
        self.check_security_server = check_security


class Passenger:
    def __init__(self, id):
        self.id = id
        self.arrival_time = 0
        self.check_in_start_waiting_time = 0
        self.check_in_waiting_time = 0
        self.check_in_time = 0
        self.check_info_start_waiting_time = 0
        self.check_info_waiting_time = 0
        self.check_info_time = 0
        self.check_security_start_waiting_time = 0
        self.check_security_waiting_time = 0
        self.check_security_time = 0
        self.check_special_security_start_waiting_time = 0
        self.check_special_security_waiting_time = 0
        self.check_special_security_time = 0
        self.is_finished = False

    def checkInProc(self, env, server, check_in_id):
        self.check_in_start_waiting_time = env.now
        # print(
        #     f"Passenger {self.id} start waiting at CHECK IN at {self.check_in_start_waiting_time}"
        # )
        with server.check_in_server[check_in_id - 1].resource.request() as req:
            yield req
            self.check_in_waiting_time = (
                env.now - self.check_in_start_waiting_time + self.check_in_waiting_time
            )
            # print(f"Passenger {self.id} WAIT {self.check_in_waiting_time} at CHECK IN")
            yield env.process(
                server.check_in_server[check_in_id - 1].doService(env, self.id)
            )
            # print(f"Passenger {self.id} finish CHECK IN at {env.now}")

            prob = np.random.uniform()
            # print(f"Passenger {self.id} prob {prob}")
            if prob < p[3][4]:
                env.process(self.checkInfoProc(env, server, 1))
            else:
                env.process(self.checkInfoProc(env, server, 2))

    def checkInfoProc(self, env, server, check_info_id):
        global record
        self.check_info_start_waiting_time = env.now
        # print(
        #     f"Passenger {self.id} start waiting at CHECK INFO at {self.check_info_start_waiting_time}"
        # )
        with server.check_info_server[check_info_id - 1].resource.request() as req:
            yield req
            self.check_info_waiting_time = (
                env.now
                - self.check_info_start_waiting_time
                + self.check_info_waiting_time
            )
            # print(
            #     f"Passenger {self.id} WAIT {self.check_info_waiting_time} at CHECK INFO"
            # )
            yield env.process(
                server.check_info_server[check_info_id - 1].doService(env, self.id)
            )
            # print(f"Passenger {self.id} finish CHECK INFO at {env.now}")

            if check_info_id == 1:
                prob = np.random.uniform()
                # print(f"Passenger {self.id} prob {prob}")
                if prob < p[4][6]:
                    env.process(self.checkSecurityProc(env, server, 1))
                elif prob < sum(p[4][6:8]):
                    env.process(self.checkSecurityProc(env, server, 2))
                elif prob < sum(p[4][6:9]):
                    env.process(self.checkSecurityProc(env, server, 3))
                elif prob < sum(p[4][6:10]):
                    env.process(self.checkSecurityProc(env, server, 4))
                elif prob < sum(p[4][6:11]):
                    env.process(self.checkSecurityProc(env, server, 5))
                elif prob < sum(p[4][6:12]):
                    env.process(self.checkSecurityProc(env, server, 6))
                elif prob < p[4][1] + sum(p[4][6:12]):
                    env.process(self.checkInProc(env, server, 1))
                elif prob < p[4][1] + p[4][2] + sum(p[4][6:12]):
                    env.process(self.checkInProc(env, server, 2))
                elif prob < p[4][1] + p[4][2] + p[4][3] + sum(p[4][6:12]):
                    env.process(self.checkInProc(env, server, 3))
                else:
                    print(f"Passenger {self.id} EXIT at {env.now}")
                    record.add_record(self)
            if check_info_id == 2:
                prob = np.random.uniform()
                # print(f"Passenger {self.id} prob {prob}")
                if prob < p[5][12]:
                    env.process(self.checkSecurityProc(env, server, 7))
                elif prob < p[5][1]:
                    env.process(self.checkInProc(env, server, 1))
                elif prob < p[5][2]:
                    env.process(self.checkInProc(env, server, 2))
                elif prob < p[5][3]:
                    env.process(self.checkInProc(env, server, 3))
                else:
                    print(f"Passenger {self.id} EXIT at {env.now}")
                    record.add_record(self)

    def checkSecurityProc(self, env, server, check_security_id):
        global record
        self.check_security_start_waiting_time = env.now
        # print(
        #     f"Passenger {self.id} start waiting at CHECK SECURITY-{check_security_id-1} at {self.check_security_start_waiting_time}"
        # )
        with server.check_security_server[
            check_security_id - 1
        ].resource.request() as req:
            yield req
            self.check_security_waiting_time = (
                env.now
                - self.check_security_start_waiting_time
                + self.check_security_waiting_time
            )
            # print(
            #     f"Passenger {self.id} WAIT {self.check_security_waiting_time} at CHECK SECURITY"
            # )
            yield env.process(
                server.check_security_server[check_security_id - 1].doService(
                    env, self.id
                )
            )
            # print(f"Passenger {self.id} finish CHECK SECURITY at {env.now}")

            if check_security_id < 8:
                prob = np.random.uniform()
                # print(f"Passenger {self.id} prob {prob}")
                if prob < p[6][13]:
                    env.process(self.checkSecurityProc(env, server, 8))
                else:
                    self.is_finished = True
                    print(f"Passenger {self.id} DONE at {env.now}")
                    record.add_record(self)

            if check_security_id == 8:
                prob = np.random.uniform()
                # print(f"Passenger {self.id} prob {prob}")
                if prob < p[13][0]:
                    print(f"Passenger {self.id} EXIT at {env.now}")
                    record.add_record(self)
                else:
                    self.is_finished = True
                    print(f"Passenger {self.id} DONE at {env.now}")
                    record.add_record(self)


class PassengerRecords:
    def __init__(self):
        self.record_count = 0
        self.arrival_time = np.array([])
        self.check_in_start_waiting_time = np.array([])
        self.check_in_waiting_time = np.array([])
        self.check_in_time = np.array([])
        self.check_info_start_waiting_time = np.array([])
        self.check_info_waiting_time = np.array([])
        self.check_info_time = np.array([])
        self.check_security_start_waiting_time = np.array([])
        self.check_security_waiting_time = np.array([])
        self.check_security_time = np.array([])
        self.totalTime = np.array([])
        self.passengerRecords = np.array([], dtype=object)

    def add_record(self, passenger: Passenger):
        self.passengerRecords = np.append(self.passengerRecords, passenger)
        self.record_count = self.record_count + 1
        self.arrival_time = np.append(self.arrival_time, passenger.arrival_time)

        self.check_in_start_waiting_time = np.append(
            self.check_in_start_waiting_time, passenger.check_in_start_waiting_time
        )
        self.check_in_waiting_time = np.append(
            self.check_in_waiting_time, passenger.check_in_waiting_time
        )
        self.check_in_time = np.append(self.check_in_time, passenger.check_in_time)

        self.check_info_start_waiting_time = np.append(
            self.check_info_start_waiting_time, passenger.check_info_start_waiting_time
        )
        self.check_info_waiting_time = np.append(
            self.check_info_waiting_time, passenger.check_info_waiting_time
        )
        self.check_info_time = np.append(
            self.check_info_time, passenger.check_info_time
        )

        self.check_security_start_waiting_time = np.append(
            self.check_security_start_waiting_time,
            passenger.check_security_start_waiting_time,
        )
        self.check_security_waiting_time = np.append(
            self.check_security_waiting_time, passenger.check_security_waiting_time
        )
        self.check_security_time = np.append(
            self.check_security_time, passenger.check_security_time
        )

        self.totalTime = np.append(
            self.totalTime,
            passenger.check_in_waiting_time
            + passenger.check_in_time
            + passenger.check_info_waiting_time
            + passenger.check_info_time
            + passenger.check_security_waiting_time
            + passenger.check_security_time,
        )

    def getRecordsInHour(self, startOfHour, endOfHour, sort=False, field=None):
        selected_arrival_time = np.array([])
        selected_check_in_waiting_time = np.array([])
        selected_check_in_time = np.array([])
        selected_check_info_waiting_time = np.array([])
        selected_check_info_time = np.array([])
        selected_check_security_waiting_time = np.array([])
        selected_check_security_time = np.array([])
        toltalTime = np.array([])
        for i in range(self.record_count):
            passengerRealTimeInHour = self.arrival_time[i] / 60
            if (
                passengerRealTimeInHour >= startOfHour
                and passengerRealTimeInHour <= endOfHour
            ):
                selected_arrival_time = np.append(
                    selected_arrival_time, self.arrival_time[i]
                )
                selected_check_in_waiting_time = np.append(
                    selected_check_in_waiting_time, self.check_in_waiting_time[i]
                )
                selected_check_in_time = np.append(
                    selected_check_in_time, self.check_in_time[i]
                )
                selected_check_info_waiting_time = np.append(
                    selected_check_info_waiting_time, self.check_info_waiting_time[i]
                )
                selected_check_info_time = np.append(
                    selected_check_info_time, self.check_info_time[i]
                )
                selected_check_security_waiting_time = np.append(
                    selected_check_security_waiting_time,
                    self.check_security_waiting_time[i],
                )
                selected_check_security_time = np.append(
                    selected_check_security_time, self.check_security_time[i]
                )
                toltalTime = np.append(toltalTime, self.totalTime)
        if sort is True:
            sorted_indices = np.array([])
            if field == "arrival_time":
                sorted_indices = np.argsort(selected_arrival_time)

            selected_arrival_time = selected_arrival_time[sorted_indices]
            selected_check_in_waiting_time = selected_check_in_waiting_time[
                sorted_indices
            ]
            selected_check_in_time = selected_check_in_time[sorted_indices]
            selected_check_info_waiting_time = selected_check_info_waiting_time[
                sorted_indices
            ]
            selected_check_info_time = selected_check_info_time[sorted_indices]
            selected_check_security_waiting_time = selected_check_security_waiting_time[
                sorted_indices
            ]
            selected_check_security_time = selected_check_security_time[sorted_indices]
            toltalTime = toltalTime[sorted_indices]

        return (
            selected_arrival_time,
            selected_check_in_waiting_time,
            selected_check_in_time,
            selected_check_info_waiting_time,
            selected_check_info_time,
            selected_check_security_waiting_time,
            selected_check_info_time,
            toltalTime,
        )

class StatisticsAnalyzer:
    def __init__(self, max_size):
        self.rawDataPointBuffer = np.array([])
        self.averageDataBuffer = np.array([])
        self.stdBuffer = np.array([])
        self.max_size = max_size

    def addValue(self, newval):
        self.rawDataPointBuffer = np.append(self.rawDataPointBuffer, newval)
        self.averageDataBuffer = np.append(
            self.averageDataBuffer, np.mean(self.rawDataPointBuffer)
        )
        self.stdBuffer = np.append(self.stdBuffer, np.std(self.rawDataPointBuffer))

    def addDataset(self, dataset):
        data_count = len(dataset)
        for i in range(data_count):
            self.addValue(dataset[i])

    def classify(self):
        return np.unique(self.rawDataPointBuffer, return_counts=True)

    def getAverage(self):
        return self.averageDataBuffer[len(self.averageDataBuffer) - 1]

    def getStandardDeviation(self):
        return self.stdBuffer[len(self.stdBuffer) - 1]


class CoStatisticsAnalyzer:
    def __init__(self, analyzer1: StatisticsAnalyzer, analyzer2: StatisticsAnalyzer):
        self.analyzer1 = analyzer1
        self.analyzer2 = analyzer2
        self.coVarianceBuff = np.array([])

    def getCoVariance(self):
        self.coVarianceBuff = np.array([])
        data_len = len(self.analyzer1.rawDataPointBuffer)
        for i in range(data_len):
            newcov = np.cov(
                self.analyzer1.rawDataPointBuffer, self.analyzer2.rawDataPointBuffer
            )
            self.coVarianceBuff = np.append(self.coVarianceBuff, newcov)

        return self.coVarianceBuff[len(self.coVarianceBuff) - 1]

class PassengerGenerator:
    def __init__(self, env, server):
        self.server = server
        env.process(self.generate(env))

    def generate(self, env):
        print("LAM:",str(LAM))
        i = 1
        while True:
            global static_analyzer_arrival_rate,static_analyzer_interval
            duration = random.expovariate(LAM)
            # static_analyzer_arrival_rate.addValue(1/duration)
            static_analyzer_interval.addValue(duration)
            yield env.timeout(duration)
            passenger = Passenger(i)
            passenger.arrival_time = env.now
            print(f"Passenger {i} arrive at {passenger.arrival_time}")

            prob = np.random.uniform()
            # print(f"Passenger {i} prob {prob}")
            if prob < 0.33:
                env.process(passenger.checkInProc(env, self.server, 1))
            elif prob < 0.66:
                env.process(passenger.checkInProc(env, self.server, 2))
            else:
                env.process(passenger.checkInProc(env, self.server, 3))
            i += 1

static_analyzer_arrival_rate = StatisticsAnalyzer(1000)
static_analyzer_interval=StatisticsAnalyzer(1000)
random.seed(42)

record = PassengerRecords()
env = simpy.Environment()
check_in = []
check_info = []
check_security = []
for i in range(NUMBER_CHECK_IN):
    mu_value = i + 1
    check_in.append(CheckIn(env, i, locals()[f"MU{mu_value}"]))
for i in range(NUMBER_CHECK_INFO):
    mu_value = i + 1 + NUMBER_CHECK_IN
    number_of_server = NUMBER_SERVER_OF_EACH_CHECK_INFO
    if mu_value == 5:
        number_of_server = 1
    check_info.append(CheckInfo(env, i, locals()[f"MU{mu_value}"], number_of_server))
for i in range(NUMBER_CHECK_SECURITY):
    mu_value = i + 1 + NUMBER_CHECK_IN + NUMBER_CHECK_INFO
    check_security.append(CheckSecurity(env, i, locals()[f"MU{mu_value}"]))
server = Server(check_in, check_info, check_security)
passenger_generator = PassengerGenerator(env, server)
env.run(until=SIM_TIME)


(
    selected_arrival_time,
    selected_check_in_waiting_time,
    selected_check_in_time,
    selected_check_info_waiting_time,
    selected_check_info_time,
    selected_check_security_waiting_time,
    selected_check_info_time,
    toltalTime,
) = record.getRecordsInHour(0, 10, True, "arrival_time")



print(len(selected_arrival_time))

print(
    f"Cov waiting time: {np.cov(selected_arrival_time, selected_check_in_waiting_time + selected_check_info_waiting_time + selected_check_security_waiting_time)[0,1]}"
)
print(
    f"Mean waiting time: {np.mean(selected_check_in_waiting_time + selected_check_info_waiting_time + selected_check_security_waiting_time)}"
)

static1 = StatisticsAnalyzer(1000)
static2 = StatisticsAnalyzer(1000)
static3 = StatisticsAnalyzer(1000)
static1.addDataset(selected_check_in_waiting_time)
static2.addDataset(selected_check_info_waiting_time)
static3.addDataset(selected_check_security_waiting_time)
print((static1.getAverage()+static2.getAverage()+static3.getAverage()))
# plt.show()
# plt.plot(
#     selected_arrival_time,
#     selected_check_in_waiting_time
#     + selected_check_info_waiting_time
#     + selected_check_security_waiting_time,
# )
# plt.xlabel("Arrival time")
# plt.ylabel("Total waiting time")
# plt.title("Simple Plot")
# plt.show()

# plt.plot(
#     selected_arrival_time,
#     selected_check_in_waiting_time
# )
# plt.xlabel("Arrival time")
# plt.ylabel("Total waiting time")
# plt.title("Simple Plot")
# plt.show()

# for data in static_analyzer.rawDataPointBuffer:
#     print(data)

# plt.plot(
#     static_analyzer.rawDataPointBuffer
# )
# plt.show()

# print(static_analyzer_arrival_rate.getAverage())
# print(static_analyzer_interval.getAverage())