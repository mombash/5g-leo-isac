#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

extern "C" void sample_wave(const std::complex<float>* source, int64_t source_count,
                             const double* coordinates, int64_t count,
                             const float* table, int phases, int taps,
                             std::complex<float>* output) {
    #pragma omp parallel for num_threads(8)
    for (int64_t sample = 0; sample < count; ++sample) {
        double coordinate = coordinates[sample];
        int64_t integer = static_cast<int64_t>(std::floor(coordinate));
        double phase = (coordinate-integer)*phases;
        int first_phase = static_cast<int>(phase);
        float fraction = static_cast<float>(phase-first_phase);
        std::complex<float> value(0,0);
        for (int tap = 0; tap < taps; ++tap) {
            int64_t index = integer+tap-(taps/2-1);
            if (index>=0 && index<source_count) {
                float coefficient = table[first_phase*taps+tap]*(1-fraction)+table[(first_phase+1)*taps+tap]*fraction;
                value += source[index]*coefficient;
            }
        }
        output[sample] = value;
    }
}

extern "C" void track_power(const float* maps, int blocks, int frequencies, int delays,
                             const double* times, const double* frequency_values,
                             double frequency_step, double acceleration_hz_s,
                             double wavelength_samples, float* output) {
    #pragma omp parallel for num_threads(8)
    for (int frequency = 0; frequency < frequencies; ++frequency) {
        float* result = output+frequency*delays;
        for (int delay = 0; delay<delays; ++delay) result[delay] = 0;
        double midpoint_frequency = frequency_values[frequency];
        for (int block = 0; block<blocks; ++block) {
            double time = times[block];
            double offset = -wavelength_samples*(midpoint_frequency*time+.5*acceleration_hz_s*time*time);
            int integer = static_cast<int>(std::floor(offset));
            float delay_fraction = static_cast<float>(offset-integer);
            double shifted_frequency = frequency+acceleration_hz_s*time/frequency_step;
            int lower_frequency = static_cast<int>(std::floor(shifted_frequency));
            float frequency_fraction = static_cast<float>(shifted_frequency-lower_frequency);
            for (int delay = 0; delay<delays; ++delay) {
                int lower_delay = delay+integer;
                if (lower_frequency<0 || lower_frequency+1>=frequencies || lower_delay<0 || lower_delay+1>=delays) {
                    result[delay] = -std::numeric_limits<float>::infinity();
                } else {
                    int64_t address = (static_cast<int64_t>(block)*frequencies+lower_frequency)*delays+lower_delay;
                    float low = maps[address]*(1-delay_fraction)+maps[address+1]*delay_fraction;
                    float high = maps[address+delays]*(1-delay_fraction)+maps[address+delays+1]*delay_fraction;
                    result[delay] += low*(1-frequency_fraction)+high*frequency_fraction;
                }
            }
        }
    }
}

extern "C" void keystone_rows(const std::complex<float>* input, int rows, int columns,
                               const double* scales, const float* table, int phases,
                               std::complex<float>* output) {
    double midpoint = (columns-1)/2.0;
    #pragma omp parallel for num_threads(8)
    for (int row = 0; row<rows; ++row) {
        const std::complex<float>* source = input+static_cast<int64_t>(row)*columns;
        std::complex<float>* destination = output+static_cast<int64_t>(row)*columns;
        for (int column = 0; column<columns; ++column) {
            double coordinate = midpoint+(column-midpoint)*scales[row];
            int integer = static_cast<int>(std::floor(coordinate));
            double phase = (coordinate-integer)*phases;
            int first_phase = static_cast<int>(phase);
            float fraction = static_cast<float>(phase-first_phase);
            std::complex<float> value(0,0);
            for (int tap = 0; tap<8; ++tap) {
                int index = integer+tap-3;
                if (index>=0 && index<columns) {
                    float coefficient = table[first_phase*8+tap]*(1-fraction)+table[(first_phase+1)*8+tap]*fraction;
                    value += source[index]*coefficient;
                }
            }
            destination[column] = value;
        }
    }
}
