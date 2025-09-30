(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        yellow - block
        blue - block
        purple - block
        green - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on yellow orange)
        (clear yellow)
        (ontable blue)
        (clear blue)
        (ontable purple)
        (on green purple)
        (on red green)
        (clear red)
        ; orange -> yellow
        ; blue
        ; purple -> green -> red
    )
    (:goal (and (on blue orange)
        (on red blue)
        (on purple red)
        (on green purple)
        (on yellow green)))
)
    