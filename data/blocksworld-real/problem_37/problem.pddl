(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        purple - block
        orange - block
        blue - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on purple red)
        (on orange purple)
        (on blue orange)
        (on yellow blue)
        (on green yellow)
        (clear green)
        ; red -> purple -> orange -> blue -> yellow -> green
    )
    (:goal (and (on orange red)
        (on purple orange)
        (on yellow purple)
        (on blue yellow)
        (on green blue)))
)
    