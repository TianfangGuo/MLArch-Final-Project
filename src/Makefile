CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Source files
SRCS = main.cpp simulator.cpp cpu.cpp accelerator.cpp utils.cpp
OBJS = $(SRCS:.cpp=.o)

# Output executable
TARGET = simulator

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
