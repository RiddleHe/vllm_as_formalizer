(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        orange - block
        red - block
        purple - block
        green - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on orange blue)
        (on red orange)
        (clear red)
        (ontable purple)
        (on green purple)
        (on yellow green)
        (clear yellow)
        ; blue -> orange -> red
        ; purple -> green -> yellow
    )
    (:goal (and (on yellow orange)
        (on red yellow)
        (on blue purple)
        (on green blue)))
)
    