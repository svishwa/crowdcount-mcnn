% source: http://sachinashanbhag.blogspot.de/2012/09/setting-up-random-number-generator-seed.html

function rng(x)
  randn("seed",x)
  rand("seed",x)
end
