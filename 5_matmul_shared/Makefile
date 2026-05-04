NVCC := nvcc
TARGET := matmul_shared
SRC := matmul_shared.cu
NVCCFLAGS := -O2

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
