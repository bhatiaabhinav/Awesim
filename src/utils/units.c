#include "utils.h"

//
// Unit constructors
//

Meters meters(double value) { return value; }
Seconds seconds(double value) { return value; }
MetersPerSecond mps(double value) { return value; }
MetersPerSecondSquared mpss(double value) { return value; }

//
// Distance conversions
//

Meters from_feet(double value) { return value * 0.3048; }
double to_feet(Meters m) { return m / 0.3048; }

Meters from_miles(double value) { return value * 1609.34; }
double to_miles(Meters m) { return m / 1609.34; }

Meters from_kilometers(double value) { return value * 1000.0; }
double to_kilometers(Meters m) { return m / 1000.0; }

Meters from_inches(double value) { return value * 0.0254; }
double to_inches(Meters m) { return m / 0.0254; }

Meters from_centimeters(double value) { return value * 0.01; }
double to_centimeters(Meters m) { return m / 0.01; }

Meters from_millimeters(double value) { return value * 0.001; }
double to_millimeters(Meters m) { return m / 0.001; }

//
// Speed conversions
//

MetersPerSecond from_kmph(double value) { return value * (1000.0 / 3600.0); }
double to_kmph(MetersPerSecond mps) { return mps * (3600.0 / 1000.0); }

MetersPerSecond from_mph(double value) { return value * (1609.34 / 3600.0); }
double to_mph(MetersPerSecond mps) { return mps * (3600.0 / 1609.34); }

MetersPerSecond from_fps(double value) { return value * 0.3048; }
double to_fps(MetersPerSecond mps) { return mps / 0.3048; }