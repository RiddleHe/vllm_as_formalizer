(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        orange - block
        purple - block
        green - block
        blue - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (clear red)
        (ontable orange)
        (on purple orange)
        (on green purple)
        (clear green)
        (ontable blue)
        (on yellow blue)
        (clear yellow)
        ; red
        ; orange -> purple -> green
        ; blue -> yellow
    )
    (:goal (and (on blue yellow)
        (on green blue)
        (on purple green)
        (on orange red)))
)
    