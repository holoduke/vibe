#!/usr/bin/perl
# Split a .cpp into several: each output gets the source's prelude (lines
# 1..P, normally includes + "namespace x {") plus the given line ranges and a
# closing line. usage: split_cpp.pl SRC P "closing" out.cpp=a-b,c-d:"description" ...
use strict; my ($src,$P,$close) = splice(@ARGV,0,3); open F,"<:raw",$src or die; my @l=<F>; close F;
for my $spec (@ARGV) {
  my ($o,$rest) = split /=/, $spec, 2; my ($rg,$d) = split /:/, $rest, 2;
  open O,">:raw",$o or die; (my $b = $o) =~ s/.*\///;
  print O "// $b - $d\n// (split out of ".($src =~ s/.*\///r)."; see docs/ARCHITECTURE.md for the map.)\n";
  print O @l[0..$P-1];
  for my $r (split /,/, $rg) { my ($a,$e) = split /-/, $r; print O "\n", @l[$a-1..$e-1]; }
  print O "\n$close\n"; close O; print "$o\n";
}
