import simpy
import numpy as np
import matplotlib.pyplot as plt
import random
from random import seed
from threading import Thread, Lock

mutex = Lock()

time_scale = 60

LAM = 120000

MU1 = 50000
MU2 = 50000
MU3 = 50000
MU4 = 50000

MU5 = 120000
MU6 = 30000

MU7 = 25000
MU8 = 25000
MU9 = 25000
MU10 = 25000
MU11 = 25000
MU12 = 25000
MU13 = 25000
MU14 = 25000

MU15 = 5000

NUMBER_CHECK_IN = 4
NUMBER_CHECK_INFO = 2
NUMBER_SERVER_OF_EACH_CHECK_INFO = 4
NUMBER_CHECK_SECURITY = 9
SIM_TIME = 10

p = [[0 for j in range(14)] for i in range(14)]

p[0][1] = 0.25
p[0][2] = 0.25
p[0][3] = 0.25
p[0][4] = 0.25

p[1][4] = 0.9
p[1][5] = 0.1

p[2][4] = 0.9
p[2][5] = 0.1

p[3][4] = 0.9
p[3][5] = 0.1

p[4][4] = 0.9
p[4][5] = 0.1

p[4][1] = 0.01
p[4][2] = 0.01
p[4][3] = 0.01
p[4][4] = 0.01

p[4][6] = 0.1328
p[4][7] = 0.1328
p[4][8]= 0.1328
p[4][9]= 0.1328
p[4][10] = 0.1328
p[4][11] = 0.1328
p[4][12] = 0.1328
p[4][0] = 0.03

p[5][1] = 0.01
p[5][2] = 0.01
p[5][3] = 0.01
p[5][4] = 0.01
p[5][12] = 0.9
p[5][0] = 0.06

p[6][13] = 0.02
# p6_pass = 0.1

p[7][13] = 0.02
# p7_pass = 0.1

p[8][13] = 0.02
# p8_pass = 0.1

p[9][13 ]= 0.02
# p9_pass = 0.1

p[10][13] = 0.02
# p10_pass = 0.1

p[11][13] = 0.02
# p11_pass = 0.1

p[12][13] = 0.02
# p12_pass = 0.1

p[13][13] = 0.02

p[13][0] = 0.02
# p13_pass = 0.1

def addJob():
    global num_job, num_job_current, job_env_add
    while True:
        yield job_env_add
        with mutex:
            num_job_current += 1
            static_analyzer_passenger.addValue(num_job_current)

def popJob():
    global num_job, num_job_current, job_env_pop
    while True:
        yield job_env_pop
        with mutex:
            num_job_current -= 1
            static_analyzer_passenger.addValue(num_job_current)

class CheckIn:
    def __init__(self, env, id, mu, number_of_server=1):
        self.id = id
        self.mu = mu
        self.resource = simpy.Resource(env, number_of_server)

    def doService(self, env, passenger):
        service_time = np.random.exponential(1.0 / self.mu)
        passenger.check_in_time = service_time + passenger.check_in_time
        yield env.timeout(service_time)
        # print(f"Passenger {passenger_id} finish check in at {env.now}")


class CheckInfo:
    def __init__(self, env, id, mu, number_of_server=1):
        self.id = id
        self.mu = mu
        self.resource = simpy.Resource(env, number_of_server)

    def doService(self, env, passenger):
        service_time = np.random.exponential(1.0 / self.mu)
        passenger.check_info_time = service_time + passenger.check_info_time
        yield env.timeout(service_time)
        # print(f"Passenger {passenger_id} finish check information at {env.now}")


class CheckSecurity:
    def __init__(self, env, id, mu, number_of_server=1):
        self.id = id
        self.mu = mu
        self.resource = simpy.Resource(env, number_of_server)

    def doService(self, env, passenger):
        service_time = np.random.exponential(1.0 / self.mu)
        passenger.check_security_time = service_time + passenger.check_security_time
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
        self.check_in_waiting_time = 0
        self.check_in_time = 0
        self.check_info_waiting_time = 0
        self.check_info_time = 0
        self.check_security_start_waiting_time = 0
        self.check_security_waiting_time = 0
        self.check_security_time = 0
        self.check_special_security_waiting_time = 0
        self.check_special_security_time = 0
        self.is_finished = False

    def checkInProc(self, env, server, check_in_id):
        global job_env_pop
        check_in_start_waiting_time = env.now
        # print(
        #     f"Passenger {self.id} start waiting at CHECK IN at {self.check_in_start_waiting_time}"
        # )
        with server.check_in_server[check_in_id - 1].resource.request() as req:
            yield req
            self.check_in_waiting_time = (
                env.now - check_in_start_waiting_time + self.check_in_waiting_time
            )
            # print(f"Passenger {self.id} WAIT {self.check_in_waiting_time} at CHECK IN")
            yield env.process(
                server.check_in_server[check_in_id - 1].doService(env, self)
            )
            # print(f"Passenger {self.id} finish CHECK IN at {env.now}")

            prob = np.random.uniform()
            # print(f"Passenger {self.id} prob {prob}")
            if prob < p[3][4]:
                env.process(self.checkInfoProc(env, server, 1))
            else:
                env.process(self.checkInfoProc(env, server, 2))

    def checkInfoProc(self, env, server, check_info_id):
        global record, job_env_pop
        check_info_start_waiting_time = env.now
        # print(
        #     f"Passenger {self.id} start waiting at CHECK INFO at {self.check_info_start_waiting_time}"
        # )
        with server.check_info_server[check_info_id - 1].resource.request() as req:
            yield req
            self.check_info_waiting_time = (
                env.now
                - check_info_start_waiting_time
                + self.check_info_waiting_time
            )
            # print(
            #     f"Passenger {self.id} WAIT {self.check_info_waiting_time} at CHECK INFO"
            # )
            yield env.process(
                server.check_info_server[check_info_id - 1].doService(env, self)
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
                elif prob < sum(p[4][6:13]):
                    env.process(self.checkSecurityProc(env, server, 7))
                elif prob < p[4][1] + sum(p[4][6:13]):
                    env.process(self.checkInProc(env, server, 1))
                elif prob < p[4][1] + p[4][2] + sum(p[4][6:13]):
                    env.process(self.checkInProc(env, server, 2))
                elif prob < p[4][1] + p[4][2] + p[4][3] + sum(p[4][6:13]):
                    env.process(self.checkInProc(env, server, 3))
                elif prob < p[4][1] + p[4][2] + p[4][3] + p[4][4] + sum(p[4][6:13]):
                    env.process(self.checkInProc(env, server, 4))
                else:
                    # print(f"Passenger {self.id} EXIT at {env.now}")
                    job_env_pop.succeed()
                    job_env_pop = env.event()
                    record.updatePassenger(self)
            if check_info_id == 2:
                prob = np.random.uniform()
                # print(f"Passenger {self.id} prob {prob}")
                if prob < p[5][13]:
                    env.process(self.checkSecurityProc(env, server, 8))
                elif prob < p[5][13] + p[5][1]:
                    env.process(self.checkInProc(env, server, 1))
                elif prob < p[5][13] + p[5][1] + p[5][2]:
                    env.process(self.checkInProc(env, server, 2))
                elif prob < p[5][13] + p[5][1] + p[5][2] + p[5][3]:
                    env.process(self.checkInProc(env, server, 3))
                elif prob < p[5][13] + p[5][1] + p[5][2] + p[5][3] + p[5][4]:
                    env.process(self.checkInProc(env, server, 4))
                else:
                    # print(f"Passenger {self.id} EXIT at {env.now}")
                    job_env_pop.succeed()
                    job_env_pop = env.event()
                    record.updatePassenger(self)

    def checkSecurityProc(self, env, server, check_security_id):
        global record, job_env_pop
        check_security_start_waiting_time = env.now
        # print(
        #     f"Passenger {self.id} start waiting at CHECK SECURITY-{check_security_id-1} at {self.check_security_start_waiting_time}"
        # )
        with server.check_security_server[
            check_security_id - 1
        ].resource.request() as req:
            yield req
            self.check_security_waiting_time = (
                env.now
                - check_security_start_waiting_time
                + self.check_security_waiting_time
            )
            # print(
            #     f"Passenger {self.id} WAIT {self.check_security_waiting_time} at CHECK SECURITY"
            # )
            yield env.process(
                server.check_security_server[check_security_id - 1].doService(
                    env, self
                )
            )
            # print(f"Passenger {self.id} finish CHECK SECURITY at {env.now}")

            if check_security_id < 9:
                prob = np.random.uniform()
                # print(f"Passenger {self.id} prob {prob}")
                if prob < p[6][13]:
                    env.process(self.checkSecurityProc(env, server, 9))
                else:
                    # print(f"Passenger {self.id} DONE at {env.now}")
                    job_env_pop.succeed()
                    job_env_pop = env.event()
                    record.updatePassenger(self)

            if check_security_id == 9:
                prob = np.random.uniform()
                # print(f"Passenger {self.id} prob {prob}")
                if prob < p[13][0]:
                    # print(f"Passenger {self.id} EXIT at {env.now}")
                    job_env_pop.succeed()
                    job_env_pop = env.event()
                    record.updatePassenger(self)
                else:
                    # print(f"Passenger {self.id} DONE at {env.now}")
                    job_env_pop.succeed()
                    job_env_pop = env.event()
                    record.updatePassenger(self)


class PassengerRecords:
    def __init__(self):
        self.record_count = 0
        self.arrival_time = np.array([])
        self.check_in_waiting_time = np.array([])
        self.check_in_time = np.array([])
        self.check_info_waiting_time = np.array([])
        self.check_info_time = np.array([])
        self.check_security_waiting_time = np.array([])
        self.check_security_time = np.array([])
        self.totalTime = np.array([])
        self.is_finished = np.array([], dtype=bool)
        self.passengerRecords = np.array([], dtype=object)
    
    def updatePassenger(self, passenger: Passenger):
        self.arrival_time[passenger.id] = passenger.arrival_time
        self.check_in_waiting_time[passenger.id] = passenger.check_in_waiting_time
        self.check_in_time[passenger.id] = passenger.check_in_time
        self.check_info_waiting_time[passenger.id] = passenger.check_in_waiting_time
        self.check_info_time[passenger.id] = passenger.check_info_time
        self.check_security_waiting_time[passenger.id] = passenger.check_security_waiting_time
        self.check_security_time[passenger.id] = passenger.check_security_time
        self.totalTime[passenger.id] = \
        passenger.check_in_waiting_time \
        + passenger.check_in_time \
        + passenger.check_info_waiting_time \
        + passenger.check_info_time \
        + passenger.check_security_waiting_time \
        + passenger.check_security_time
        self.is_finished[passenger.id] = True
        self.passengerRecords[passenger.id] = passenger 

    def addRecord(self, passenger: Passenger):
        self.passengerRecords = np.append(self.passengerRecords, passenger)
        self.record_count = self.record_count + 1
        self.arrival_time = np.append(self.arrival_time, passenger.arrival_time)

        self.check_in_waiting_time = np.append(
            self.check_in_waiting_time, passenger.check_in_waiting_time
        )
        self.check_in_time = np.append(self.check_in_time, passenger.check_in_time)

        self.check_info_waiting_time = np.append(
            self.check_info_waiting_time, passenger.check_info_waiting_time
        )
        self.check_info_time = np.append(
            self.check_info_time, passenger.check_info_time
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

        self.is_finished = np.append(self.is_finished, False)

    def getRawRecord(self):
        return (
            self.arrival_time,
            self.check_in_waiting_time,
            self.check_in_time,
            self.check_info_waiting_time,
            self.check_info_time,
            self.check_security_waiting_time,
            self.check_info_time,
            self.totalTime,
        )

    def getRawRecordFiltered(self):
        return (
            self.arrival_time[record.is_finished],
            self.check_in_waiting_time[record.is_finished],
            self.check_in_time[record.is_finished],
            self.check_info_waiting_time[record.is_finished],
            self.check_info_time[record.is_finished],
            self.check_security_waiting_time[record.is_finished],
            self.check_info_time[record.is_finished],
            self.totalTime[record.is_finished],
        )

class StatisticsAnalyzer:
    def __init__(self):
        self.rawDataPointBuffer = np.array([])
        self.averageDataBuffer = np.array([])
        self.stdBuffer = np.array([])

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
        global job_env_add
        print("LAM:",str(LAM))
        i = 0
        while True:
            global static_analyzer_arrival_rate,static_analyzer_interval,num_job
            duration = random.expovariate(LAM/time_scale)
            yield env.timeout(duration)
            static_analyzer_interval.addValue(duration)
            passenger = Passenger(i)
            passenger.arrival_time = env.now
            job_env_add.succeed()
            job_env_add = env.event()
            record.addRecord(passenger)
            print(f"Passenger {i} arrive at {passenger.arrival_time}")
            
            prob = np.random.uniform()
            # print(f"Passenger {i} prob {prob}")
            if prob < p[0][1]:
                env.process(passenger.checkInProc(env, self.server, 1))
            elif prob < p[0][1] + p[0][2]:
                env.process(passenger.checkInProc(env, self.server, 2))
            elif prob < p[0][1] + p[0][2] + p[0][3]:
                env.process(passenger.checkInProc(env, self.server, 3))
            else:
                env.process(passenger.checkInProc(env, self.server, 4))
            i += 1

static_analyzer_passenger = StatisticsAnalyzer()
static_analyzer_interval = StatisticsAnalyzer()
num_job_current = 0
num_job = []
random.seed(42)
record = PassengerRecords()
env = simpy.Environment()
job_env_add = env.event()
job_env_pop = env.event()
check_in = []
check_info = []
check_security = []

for i in range(NUMBER_CHECK_IN):
    mu_value = i + 1
    check_in.append(CheckIn(env, i, locals()[f"MU{mu_value}"]/time_scale))
for i in range(NUMBER_CHECK_INFO):
    mu_value = i + 1 + NUMBER_CHECK_IN
    number_of_server = NUMBER_SERVER_OF_EACH_CHECK_INFO
    if mu_value == 6:
        number_of_server = 1
    check_info.append(CheckInfo(env, i, locals()[f"MU{mu_value}"]/time_scale, number_of_server))
for i in range(NUMBER_CHECK_SECURITY):
    mu_value = i + 1 + NUMBER_CHECK_IN + NUMBER_CHECK_INFO
    check_security.append(CheckSecurity(env, i, locals()[f"MU{mu_value}"]/time_scale))

server = Server(check_in, check_info, check_security)
passenger_generator = PassengerGenerator(env, server)
env.process(addJob())
env.process(popJob())
env.run(until=SIM_TIME)

(
    raw_arrival_time,
    raw_check_in_waiting_time,
    raw_check_in_time,
    raw_check_info_waiting_time,
    raw_check_info_time,
    raw_check_security_waiting_time,
    raw_check_security_time,
    toltalTime,
) = record.getRawRecord()

(
    selected_arrival_time,
    selected_check_in_waiting_time,
    selected_check_in_time,
    selected_check_info_waiting_time,
    selected_check_info_time,
    selected_check_security_waiting_time,
    selected_check_security_time,
    toltalTime,
) = record.getRawRecordFiltered()

print(f"Total job in: {len(raw_arrival_time)}")
print(f"Total job out: {len(selected_arrival_time)}")
print(f"Mean of Job: {static_analyzer_passenger.getAverage()}")
print(f"Mean total time: {np.mean(toltalTime)/time_scale}")

if True:
    # plt.hist(static_analyzer_interval.rawDataPointBuffer, 100)
    # plt.show()

    plt.plot(static_analyzer_interval.averageDataBuffer)
    plt.show()

    plt.plot(static_analyzer_passenger.averageDataBuffer)
    plt.show()

    plt.plot(
        selected_arrival_time,
        toltalTime
    )
    plt.xlabel("Arrival time")
    plt.ylabel("Total time")
    plt.show()